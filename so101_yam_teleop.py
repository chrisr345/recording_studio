#!/usr/bin/env python3
"""SO-101 leader arm to YAM follower arm teleoperation.

Reads SO-101 Feetech joint positions via lerobot and maps them to YAM DM
motor joint commands using anchor-relative delta motion in joint space.

The SO-101 has 5 arm joints (shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll) plus a gripper. The YAM has 6 arm joints (joint1–6)
plus a gripper. YAM joint6 (wrist yaw) is held at its engage-time position.

Quick start
-----------
# First run — calibrates SO-101 interactively if no calibration file exists:
    .venv-lerobot/bin/python so101_yam_teleop.py --port /dev/ttyUSB0

# Execute mode (actually moves the robot):
    .venv-lerobot/bin/python so101_yam_teleop.py --port /dev/ttyUSB0 --execute

Press ENTER   → engage/disengage (sets anchor from current poses)
Ctrl-C        → quit

Gains
-----
--gains J1 J2 J3 J4 J5  (default 1.0 each)
  Positive = same direction as SO-101. Negate a joint to reverse it.
  Magnitude = how many radians of YAM motion per radian of SO-101 motion
  (SO-101 degrees are converted to radians first, then multiplied by gain).
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Joint constants
# ---------------------------------------------------------------------------

# SO-101 motor names in the order returned by SOLeader.get_action()
SO101_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SO101_GRIPPER = "gripper"

# YAM arm joint limits from yam.xml (radians). Used for display/clipping.
YAM_JOINT_LIMITS = np.array([
    [-2.618,  3.054],   # joint1 — shoulder_pan
    [ 0.000,  3.650],   # joint2 — shoulder_lift
    [ 0.000,  3.665],   # joint3 — elbow_flex
    [-1.571,  1.571],   # joint4 — wrist_flex
    [-1.571,  1.571],   # joint5 — wrist_roll
    [-2.094,  2.094],   # joint6 — wrist_yaw (held at anchor)
])

# YAM gripper is normalized to [0, 1] (0 = closed, 1 = open).
# SO-101 gripper is normalized to [0, 100] by lerobot calibration.
SO101_GRIPPER_SCALE = 1.0 / 100.0  # maps SO-101 [0,100] → YAM [0,1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.perf_counter()


def _fmt(a: np.ndarray | None, precision: int = 3) -> str:
    if a is None:
        return "None"
    return np.array2string(np.asarray(a, dtype=float), precision=precision, suppress_small=True)


def _stdin_ready() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])


# ---------------------------------------------------------------------------
# SO-101 reader
# ---------------------------------------------------------------------------

class SO101Reader:
    """Direct FeetechMotorsBus reader — no calibration wizard required.

    For delta-based teleop we only need to measure how much each joint moves
    relative to the engage anchor. Absolute calibration is unnecessary, so we
    skip lerobot's interactive wizard entirely and read raw encoder ticks
    (0-4095, one full revolution = 4096 ticks = 2π rad).
    """

    TICKS_PER_REV: int = 4096
    MOTOR_NAMES: list[str] = [*SO101_ARM_JOINTS, SO101_GRIPPER]

    def __init__(self, port: str, **_kwargs):
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lerobot", "src"))
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors import Motor, MotorNormMode

        self._bus = FeetechMotorsBus(
            port=port,
            motors={
                name: Motor(idx + 1, "sts3215", MotorNormMode.RANGE_M100_100)
                for idx, name in enumerate(self.MOTOR_NAMES)
            },
        )
        # Ignore the latched Input Voltage Error bit (bit 0) in ping/read
        # responses — it persists from prior power events but does not affect
        # operation when supply voltage is actually fine (12 V confirmed).
        _orig = self._bus._is_error
        self._bus._is_error = lambda e: _orig(e & ~0x01)

        # handshake=False skips the firmware-version check; we still verify
        # motors exist via the patched ping above.
        self._bus.connect(handshake=False)
        self._bus.disable_torque()  # arm must move freely as a leader

    def _read_raw(self) -> dict[str, int]:
        """Sequential reads of Present_Position in raw ticks (0-4095)."""
        result = {}
        for name in self.MOTOR_NAMES:
            result[name] = self._bus.read("Present_Position", name, normalize=False)
        return result

    def read(self) -> tuple[np.ndarray, float]:
        """Returns (arm_rad[5], gripper_norm[0-1]).

        arm_rad: raw-tick angles converted to radians (monotonically increasing
        from 0 to 2π over one revolution — no homing offset applied).
        gripper_norm: gripper ticks normalised to [0, 1] over the full range.
        """
        raw = self._read_raw()
        scale = 2.0 * math.pi / self.TICKS_PER_REV
        arm = np.array([raw[j] * scale for j in SO101_ARM_JOINTS], dtype=float)
        grip = raw[SO101_GRIPPER] / (self.TICKS_PER_REV - 1)
        return arm, grip

    def close(self) -> None:
        self._bus.disconnect()


# ---------------------------------------------------------------------------
# Teleop state machine
# ---------------------------------------------------------------------------

class SO101YamTeleop:
    """Maps SO-101 joint deltas to YAM joint targets.

    State: IDLE (gravity comp on YAM) or ACTIVE (tracking SO-101 motion).
    Toggle with engage() / disengage().
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.gains = np.asarray(args.gains, dtype=float)  # shape (5,), one per arm joint
        self.gripper_invert = args.gripper_invert

        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        self.robot = get_yam_robot(
            channel=args.channel,
            gripper_type=GripperType.LINEAR_4310,
            zero_gravity_mode=False,
        )

        q0 = np.asarray(self.robot.get_joint_pos(), dtype=float)
        self.q_cmd = q0.copy()

        self._yam_anchor: np.ndarray | None = None
        self._so101_arm_anchor: np.ndarray | None = None
        self.active = False

        # Gripper uses absolute position, not delta. Track the observed
        # min/max raw values across the whole session so that the full
        # physical travel of the trigger maps to the full YAM [0,1] range.
        # These persist across engage/disengage cycles.
        self._grip_min: float = float("inf")
        self._grip_max: float = float("-inf")

    # ------------------------------------------------------------------

    def engage(self, so101_arm: np.ndarray, so101_grip: float) -> None:
        q_actual = np.asarray(self.robot.get_joint_pos(), dtype=float)
        self._yam_anchor = q_actual.copy()
        self._so101_arm_anchor = so101_arm.copy()
        self.q_cmd = q_actual.copy()
        self.active = True

    def disengage(self) -> None:
        self._yam_anchor = None
        self._so101_arm_anchor = None
        self.active = False
        if hasattr(self.robot, "enter_gravity_comp_idle"):
            self.robot.enter_gravity_comp_idle()

    def _gripper_cmd(self, so101_grip: float) -> float:
        """Map raw SO-101 trigger position to YAM gripper command [0, 1].

        Uses an auto-ranging scheme: the first time you fully open and fully
        close the trigger the range is learned, and the full travel maps to
        the full YAM gripper range. Works regardless of engage position.
        """
        self._grip_min = min(self._grip_min, so101_grip)
        self._grip_max = max(self._grip_max, so101_grip)
        grip_range = self._grip_max - self._grip_min
        if grip_range < 0.005:
            return self.q_cmd[6]  # not enough travel seen yet — hold position
        norm = (so101_grip - self._grip_min) / grip_range  # 0 = min, 1 = max
        if self.gripper_invert:
            norm = 1.0 - norm
        return float(np.clip(norm, 0.0, 1.0))

    # ------------------------------------------------------------------

    def step(self, so101_arm: np.ndarray, so101_grip: float) -> dict:
        q_actual = np.asarray(self.robot.get_joint_pos(), dtype=float)

        if not self.active:
            self._gripper_cmd(so101_grip)  # keep updating range even while idle
            return {"active": False, "q_actual": q_actual}

        # Delta from anchor in radians (SO-101 positions are raw ticks→rad),
        # scaled by per-joint gains. Gain sign controls direction.
        delta_rad = (so101_arm - self._so101_arm_anchor) * self.gains

        # Build 7-element target: [joint1..5 from delta, joint6 held, gripper]
        q_target = self._yam_anchor.copy()
        q_target[:5] = self._yam_anchor[:5] + delta_rad
        # joint6 (index 5) stays at anchor — no SO-101 joint maps to it

        # Gripper: absolute position from auto-ranged trigger reading
        q_target[6] = self._gripper_cmd(so101_grip)

        # Clip arm joints to YAM limits
        for i in range(6):
            q_target[i] = float(np.clip(q_target[i], YAM_JOINT_LIMITS[i, 0], YAM_JOINT_LIMITS[i, 1]))

        # Rate-limit: max step per tick
        if self.args.max_joint_step > 0:
            delta_cmd = q_target - self.q_cmd
            delta_cmd[:6] = np.clip(delta_cmd[:6], -self.args.max_joint_step, self.args.max_joint_step)
            q_target = self.q_cmd + delta_cmd

        self.q_cmd = q_target.copy()

        if self.args.execute:
            self.robot.command_joint_pos(self.q_cmd)

        return {
            "active": True,
            "q_actual": q_actual,
            "q_cmd": self.q_cmd.copy(),
            "delta_rad": delta_rad.copy(),
            "so101_grip": so101_grip,
            "yam_grip_cmd": float(self.q_cmd[6]),
        }

    def close(self) -> None:
        self.robot.close()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    print(f"Connecting to SO-101 on {args.port} …")
    reader = SO101Reader(port=args.port)
    print("SO-101 connected.")

    print(f"Connecting to YAM on {args.channel} …")
    teleop = SO101YamTeleop(args)
    print("YAM connected.")

    mode_str = "EXECUTE MODE — deadman key will move the robot." if args.execute else "SHADOW MODE — no motion, computing targets only."
    print(f"\n{mode_str}")
    print(f"Gains: {args.gains}  max_joint_step: {args.max_joint_step} rad")
    print("\nPress ENTER to engage/disengage teleop.  Ctrl-C to quit.\n")

    period = 1.0 / args.hz
    next_tick = _now()
    next_print = _now()

    try:
        while True:
            now = _now()
            sleep_s = next_tick - now
            if sleep_s > 0.001:
                time.sleep(sleep_s)
            next_tick = max(next_tick + period, _now())

            # Check for Enter key
            if _stdin_ready():
                sys.stdin.readline()
                so101_arm, so101_grip = reader.read()
                if teleop.active:
                    teleop.disengage()
                    print("\n[DISENGAGED] YAM in gravity comp.")
                else:
                    teleop.engage(so101_arm, so101_grip)
                    print(
                        f"\n[ENGAGED]"
                        f"\n  SO-101 anchor arm: {_fmt(so101_arm, 3)} rad"
                        f"\n  SO-101 anchor grip: {so101_grip:.3f}"
                        f"\n  YAM anchor: {_fmt(teleop._yam_anchor[:6], 3)} rad  grip={teleop._yam_anchor[6]:.3f}"
                    )

            so101_arm, so101_grip = reader.read()
            info = teleop.step(so101_arm, so101_grip)

            now = _now()
            if now >= next_print:
                grip_range = teleop._grip_max - teleop._grip_min
                grip_calibrated = grip_range >= 0.005
                if info["active"]:
                    print(
                        f"ACTIVE | so101={_fmt(so101_arm, 3)}"
                        f" | delta={_fmt(info['delta_rad'], 3)}"
                        f" | q_cmd[0:6]={_fmt(info['q_cmd'][:6])}"
                        f" | grip_raw={so101_grip:.4f}"
                        f" [range={grip_range:.4f}{'✓' if grip_calibrated else ' open+close trigger to calibrate'}]"
                        f" grip_cmd={info['yam_grip_cmd']:.3f}"
                        f" | sent={args.execute}"
                    )
                else:
                    q_act = info["q_actual"]
                    print(
                        f"IDLE   | so101={_fmt(so101_arm, 3)}"
                        f" grip_raw={so101_grip:.4f} [range={grip_range:.4f}{'✓' if grip_calibrated else ' open+close trigger'}]"
                        f" | yam={_fmt(q_act[:6])} grip={q_act[6]:.3f}"
                    )
                next_print = now + 1.0 / args.print_hz

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down …")
        reader.close()
        teleop.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SO-101 leader arm → YAM follower arm teleoperation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port", default="/dev/ttyUSB0",
        help="SO-101 USB serial port (e.g. /dev/ttyUSB0 or /dev/ttyACM0)",
    )
    parser.add_argument("--channel", default="can0", help="YAM CAN interface name")
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually send joint commands to the YAM (default: shadow/dry-run mode)",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Control loop frequency (Hz)")
    parser.add_argument("--print-hz", type=float, default=5.0, help="Status print frequency (Hz)")
    parser.add_argument(
        "--gains", type=float, nargs=5, default=[-1.0, 1.0, -1.0, -1.0, 1.0],
        metavar=("J1", "J2", "J3", "J4", "J5"),
        help=(
            "Per-joint gain for SO-101→YAM mapping. One value per arm joint "
            "(shoulder_pan→joint1, shoulder_lift→joint2, elbow_flex→joint3, "
            "wrist_flex→joint4, wrist_roll→joint5). Negate to reverse a joint."
        ),
    )
    parser.add_argument(
        "--max-joint-step", type=float, default=0.05,
        help="Max YAM arm joint change per tick (rad). Limits slew rate. 0 = disabled.",
    )
    parser.add_argument(
        "--gripper-invert", action="store_true", default=True,
        help="Invert gripper direction: trigger_max→YAM closed, trigger_min→YAM open (default: on).",
    )
    parser.add_argument(
        "--no-gripper-invert", dest="gripper_invert", action="store_false",
        help="Disable gripper inversion.",
    )
    args = parser.parse_args()

    if args.hz <= 0:
        parser.error("--hz must be positive")
    if args.print_hz <= 0:
        parser.error("--print-hz must be positive")

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
