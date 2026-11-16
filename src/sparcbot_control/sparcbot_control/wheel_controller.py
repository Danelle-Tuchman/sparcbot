"""Wheel velocity controller for both wheels. Write your controller in WheelController.compute()."""

from dataclasses import dataclass

from sparcbot_control.controller_config import WheelControllerConfig
from sparcbot_control.data_logger import DataLogger

LEFT = 0
RIGHT = 1


@dataclass
class WheelControllerOutput:
    """Result for one wheel for one control step."""

    error_m_s: float = 0.0            # OPTIONAL for debugging: target - measured
    feedforward_command: float = 0.0  # OPTIONAL for debugging: feedforward part (including deadband)
    feedback_command: float = 0.0     # OPTIONAL for debugging: feedback (PID) part
    command: float = 0.0              # duty sent to the motor, in [-1, 1]


class WheelController:
    """Velocity controller for the left and right wheels.

    Every list below is [left, right]; index with LEFT and RIGHT.
    Constants: self.config[LEFT] / self.config[RIGHT] (see controller_config.py;
    the values come from sparcbot.yaml) and self.base_width_m.
    """

    def __init__(self, left_config: WheelControllerConfig, right_config: WheelControllerConfig,
                 base_width_m: float, log_file: str = ''):
        if base_width_m <= 0.0:
            raise ValueError('base_width_m must be > 0')

        self.config = [left_config, right_config]
        self.base_width_m = base_width_m     # distance between the wheels [m]

        # State you can use in compute(). Update it there as your controller needs.
        self.time_s = 0.0                    # time since the first compute() after start/reset [s]
        self.first_step = True               # True until the first compute() after start/reset
        self.previous_error = [0.0, 0.0]     # error at the previous step [m/s]
        self.previous_command = [0.0, 0.0]   # command u sent at the previous step
        self.error_integral = [0.0, 0.0]     # running integral of the error [m]

        self._clock_started = False          # keeps time_s independent of your own state

        # CSV logger for self.log(); the file is set by `log_file` in sparcbot.yaml.
        self._logger = DataLogger(log_file) if log_file else None
        self._log_warned = False

    def diff_drive_kinematics(self, linear_m_s: float, angular_rad_s: float) -> list:
        """Differential-drive kinematics: body (v, w) -> [left, right] wheel velocities [m/s].

            v_left  = v - w * base_width / 2
            v_right = v + w * base_width / 2
        """
        half_width = 0.5 * self.base_width_m
        return [linear_m_s - angular_rad_s * half_width,
                linear_m_s + angular_rad_s * half_width]

    def log(self, **values) -> None:
        """Write one row to the log file, e.g. for measuring the deadband / feedforward:

            self.log(time_s=self.time_s,
                     left_command=u[LEFT], left_measured=measured_velocity_m_s[LEFT],
                     right_command=u[RIGHT], right_measured=measured_velocity_m_s[RIGHT])

        The column names come from the first call; keep using the same names.
        Set `log_file` in sparcbot.yaml to turn logging on.
        """
        if self._logger is None:
            if not self._log_warned:
                print("WheelController.log(): no log_file set in sparcbot.yaml; nothing is written.")
                self._log_warned = True
            return
        self._logger.write(values)

    def compute(
        self,
        linear_velocity_m_s: float,
        angular_velocity_rad_s: float,
        measured_velocity_m_s: list,
        dt_s: float,
    ) -> list:
        """Return [left_output, right_output] (WheelControllerOutput) for this step.

        Args:
            linear_velocity_m_s:    commanded forward velocity of the robot [m/s].
            angular_velocity_rad_s: commanded turn rate of the robot [rad/s], + = left.
            measured_velocity_m_s:  [left, right] measured wheel velocities [m/s].
            dt_s:                   time since the previous step [s], > 0.
        """
        if dt_s <= 0.0:
            raise ValueError('compute requires dt_s > 0')

        if self._clock_started:
            self.time_s += dt_s
        self._clock_started = True

        # [left, right] wheel velocities that produce the commanded (v, w).
        target_velocity_m_s = self.diff_drive_kinematics(
            linear_velocity_m_s, angular_velocity_rad_s)

        # TODO: write the controller for both wheels: 
        # Suggestions: you can use feedforward, deadband compensation and feedback, 
        # using target_velocity_m_s, measured_velocity_m_s, self.config[LEFT] / self.config[RIGHT]
        # and the state set up in __init__. Use self.log(...) to record data.

        # BEGIN SOLUTION

         # BEGIN SOLUTION
        outputs = [None, None]
 
        for side in (LEFT, RIGHT):
            cfg = self.config[side]
            v_des = target_velocity_m_s[side]
            v_meas = measured_velocity_m_s[side]
            error = v_des - v_meas
 
            # 1. Commanded stop: output zero and clear this wheel's memory so it doesn't creep.
            if abs(v_des) < cfg.zero_velocity_threshold:
                self.error_integral[side] = 0.0
                self.previous_error[side] = error
                self.previous_command[side] = 0.0
                outputs[side] = WheelControllerOutput(error_m_s=error, command=0.0)
                continue
 
            # 2. Feedforward: static friction + speed-proportional term, plus breakaway
            #    (deadband) compensation when the wheel is stopped.
            #    ASSUMES reverse_static / reverse_breakaway are POSITIVE in sparcbot.yaml.
            wheel_stopped = abs(v_meas) < cfg.stopped_velocity_threshold
            if v_des > 0.0:
                ff = cfg.forward_static + cfg.forward_velocity_gain * v_des
                if wheel_stopped:
                    ff = max(ff, cfg.forward_breakaway)
            else:
                ff = -cfg.reverse_static + cfg.reverse_velocity_gain * v_des
                if wheel_stopped:
                    ff = min(ff, -cfg.reverse_breakaway)
 
            # 3. P term
            p = cfg.kp * error
 
            # 4. I term with clamping anti-windup: freeze only when the last command was
            #    saturated AND the error pushes further in that same direction.
            u_prev = self.previous_command[side]
            saturated = abs(u_prev) >= cfg.max_command
            pushing_further = error * u_prev > 0.0
            if not (saturated and pushing_further):
                self.error_integral[side] += error * dt_s
            i = cfg.ki * self.error_integral[side]
 
            # 5. D term on the error (no filter). Skipped on the first step, when there is
            #    no previous error yet. Leave kd = 0 unless you need it (see notes).
            if self.first_step:
                d = 0.0
            else:
                d = cfg.kd * (error - self.previous_error[side]) / dt_s
 
            # 6. Combine and saturate to the allowed duty range.
            feedback = p + i + d
            u = ff + feedback
            u_sat = max(-cfg.max_command, min(cfg.max_command, u))
 
            # 7. Remember for next step (saturated command, for anti-windup).
            self.previous_error[side] = error
            self.previous_command[side] = u_sat
 
            outputs[side] = WheelControllerOutput(error_m_s=error,
                                                  feedforward_command=ff,
                                                  feedback_command=feedback,
                                                  command=u_sat)
 
        self.first_step = False
 
        left_wheel_output = outputs[LEFT]
        right_wheel_output = outputs[RIGHT]
 
        self.log(
            time_s=self.time_s,
            left_target=target_velocity_m_s[LEFT],
            left_measured=measured_velocity_m_s[LEFT],
            left_ff=left_wheel_output.feedforward_command,
            left_fb=left_wheel_output.feedback_command,
            left_integral=self.error_integral[LEFT],
            left_command=left_wheel_output.command,
            right_target=target_velocity_m_s[RIGHT],
            right_measured=measured_velocity_m_s[RIGHT],
            right_ff=right_wheel_output.feedforward_command,
            right_fb=right_wheel_output.feedback_command,
            right_integral=self.error_integral[RIGHT],
            right_command=right_wheel_output.command,
        )
 
        return [left_wheel_output, right_wheel_output]



        # left_wheel_output = WheelControllerOutput(command=0.0) #TODO set command to your final controller output

        # right_wheel_output = WheelControllerOutput(command=0.0) #TODO set command to your final controller output
        # return [left_wheel_output, right_wheel_output]
        # END SOLUTION

    def reset(self) -> None:
        """Clear the controller state (called when the robot stops or is reset)."""
        self.time_s = 0.0
        self._clock_started = False
        self.first_step = True
        self.previous_error = [0.0, 0.0]
        self.previous_command = [0.0, 0.0]
        self.error_integral = [0.0, 0.0]

    def close(self) -> None:
        """Flush and close the log file (called by the node on shutdown)."""
        if self._logger is not None:
            self._logger.close()