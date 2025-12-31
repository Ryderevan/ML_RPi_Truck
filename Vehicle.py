import numpy as np
import threading
import time
import smbus2
import Linear_Agent as Linear_Agent
#import DQN_Agent as DQN_Agent
import pigpio
#Testtest

class vehicle:
    def __init__(self, throttle_pin, steering_pin, servo_pin, modeswitch_pin):
        self.theta = 0 #Vehicle angle (degrees)
        self.thetaprime = 0 #Vehicle angular velocity (degrees/second)
        self.target_theta = 0  # Target heading for PID heading-hold mode; can be reset when steering direction changes
        self.timestamp = 0 
        self.throttle_input_pulse_width = 0  # Pin 23: throttle input (microseconds)
        self.steering_input_pulse_width = 0  # Pin 24: steering input (microseconds)
        self.modeswitch_input_pulse_width = 0
        self.prev_heading_error = 0.0  # Track previous error for convergence reward
        
        # Lock for protecting reads/writes to shared state updated by threads
        self._lock = threading.Lock()
        
    #Verbosity for PWM monitoring
        self._pwm_verbose = False

        # pigpio connection (single instance for all pins)
        self._pi = None
        self._pwm_callbacks = {}  # pin -> callback handle dict
        
        # Drive mode system: 1=pass-through, 2=data collection, 3=linear inference, 4=PID
        self.drive_mode = 1  # default to pass-through steering
        
        # Data collection for drive mode 2 (offline RL)
        self.data_log = []  # list of (state, action, reward) tuples
        self.data_collection_active = False
        self.data_collection_start_time = 0
        self.data_collection_duration = 5.0  # seconds
        self.throttle_threshold = 50.0  # us above neutral to trigger data collection
        self.run_counter = 0  # number of completed data collection runs
        
        # Control loop state tracking
        self.integral_error = 0.0  # PID integral term
        self.previous_steering_input = 1600.0  # Track steering input changes
        self.servo_neutral = 1600.0  # Servo centerpoint (adjust if servo is misaligned; default 1500)
        self.steering_deadband_us = 50.0  # +/- deadband around servo centerpoint
        self.brake_threshold = -50.0  # us below neutral for brake detection
        
        # External agent reference
        self.linear_agent = None
        
        # Gyroscope bias offset (calibrate when stationary)
        self._gyro_z_bias = 0  # degrees per second offset (adjust based on observed drift), calculated on 1st loop

        # Connect to pigpiod and start PWM monitors
        if pigpio is not None:
            try:
                self._pi = pigpio.pi()
                if self._pi.connected:
                    self.start_pwm_monitor(pin=throttle_pin, smoothing=0.0)  # throttle
                    self.start_pwm_monitor(pin=steering_pin, smoothing=0.0)  # steering
                    self.start_pwm_monitor(pin=modeswitch_pin, smoothing=0.0)  # mode switch
                else:
                    print("Warning: could not connect to pigpiod; PWM monitoring disabled")
                    self._pi = None
            except Exception as e:
                print(f"Warning: pigpio setup failed: {e}")
                self._pi = None
        else:
            print("Warning: pigpio not available; PWM monitoring disabled")


        # Configure servo output settings (BCM pin 18, 180 Hz)
        self._servo_pin = servo_pin
        self._servo_hz = 180
        self._using_pigpio_pwm = True

        # I2C addresses
        LSM6DSOX_ADDR = 0x6A  # IMU (gyro for theta_dot)

        # LSM6DSOX register addresses
        CTRL1_XL = 0x10  # Accelerometer control
        CTRL2_G = 0x11   # Gyroscope control
        OUTX_L_G = 0x22  # Gyroscope data start

        # Create I2C bus
        bus = smbus2.SMBus(1)

        # Enable accelerometer (104 Hz, ±2g)
        bus.write_byte_data(LSM6DSOX_ADDR, CTRL1_XL, 0x40)
        
        # Enable gyroscope (104 Hz, 250 dps)
        bus.write_byte_data(LSM6DSOX_ADDR, CTRL2_G, 0x40)

        # Start a background monitor that updates sensor/state fields at a fixed rate.
        self._running = True

        def _attitude_monitor_loop():
            last_t = time.time()
            # ensure numeric fields are initialized for the loop
            i = 0
            while self._running:
                t = time.time()
                dt = max(1e-6, t - last_t)
                last_t = t
                # Read gyroscope for theta_dot (heading rate)
                # Vehicle is rotating around Z axis for yaw
                gyro_data = bus.read_i2c_block_data(LSM6DSOX_ADDR, OUTX_L_G, 6)
                z_gyro = (gyro_data[5] << 8) | gyro_data[4]
                if z_gyro > 32767: z_gyro -= 65536
                theta_dot_raw = z_gyro * 8.75 / 1000  # 8.75 mdps/LSB measured in degrees per second
                if i == 0:
                    # First iteration, check for gyro bias
                    self._gyro_z_bias = -theta_dot_raw

                self.thetaprime = theta_dot_raw + self._gyro_z_bias  # Apply bias correction
                
                # Integrate gyro for heading (dead reckoning)
                self.theta += self.thetaprime * dt
                
                # Normalize to -180 to +180 range
                while self.theta > 180:
                    self.theta -= 360
                while self.theta < -180:
                    self.theta += 360
                
                self.timestamp = t
                i += 1
                time.sleep(0.02)  # ~50 Hz poll rate


        self._attitude_monitor_thread = threading.Thread(target=_attitude_monitor_loop, daemon=True)
        self._attitude_monitor_thread.start()
    
    def set_drive_mode(self, mode):
        """Set drive mode (1=pass-through, 2=data collection, 3=linear inference, 4=PID)"""
        self.drive_mode = mode
        print(f"Drive mode set to {mode}")
        self.print_mode_info(mode)

        # Linear model setup for mode 3 (inference only for linear model)
        if self.drive_mode == 3:
            PRETRAINED_MODEL = 'linear_model.pkl'
            self.linear_agent = Linear_Agent.LinearAgent(state_dim=2)
            
            try:
                self.linear_agent.load_model(PRETRAINED_MODEL)
                print(f"Loaded linear model: {PRETRAINED_MODEL}")
                print("Inference mode with linear regression")
            except FileNotFoundError:
                print(f"ERROR: Pre-trained model '{PRETRAINED_MODEL}' not found!")
                self.shutdown()
                exit(1)
    
    def reset_target_heading(self):
        """Reset the target heading to the current heading.
        """
        with self._lock:
            self.target_theta = self.theta
        print(f"Target heading reset to {self.target_theta:.2f}°")
    
    def _passthrough_steering(self):
        """Pass-through mode: output the current steering input directly to servo.
        """
        with self._lock:
            steering_pulse_width = self.steering_input_pulse_width
        self._write_servo_pulsewidth(steering_pulse_width)

    def _write_servo_pulsewidth(self, pulse_width_us):
        """Write a servo pulse width (in microseconds) to the steering servo on BCM pin18.
        """

        try:
            width = float(pulse_width_us)
        except Exception:
            print(f"_write_servo_pulsewidth: invalid pulse width {pulse_width_us}")
            return

        # Clamp to reasonable servo range (1000 to 2000 us to allow full steering range)
        width = max(1000, min(2000.0, width))

        if self._pi is not None:
            try:
                self._pi.set_servo_pulsewidth(self._servo_pin, width)
                if self._pwm_verbose:
                    print(f"_write_servo_pulsewidth: set servo pulse width {width:.0f} us")
            except Exception as e:
                print(f"Error writing servo pulse width: {e}")
        else:
            print(f"_write_servo_pulsewidth: pigpio not connected")
    
    def PID_action(self, obs, target_theta, integral_error, Kp=6, Kd=1, Ki=0, debug=False):
        """PID steering controller based on gyro heading feedback.
        
        When steering input is centered (within deadband), use the vehicle's theta (heading error)
        as the control error. The PID controller outputs a steering command to stabilize the heading.
        
        Args:
            obs: observation array [theta, thetaprime, steering_input_pulse_width, throttle_input_pulse_width]
                - theta: vehicle yaw angle (degrees)
                - thetaprime: vehicle yaw rate (degrees/second) - derivative of error
                - steering_input_pulse_width: receiver steering input (us)
                - throttle_input_pulse_width: receiver throttle input (us)
            target_theta: target heading setpoint (degrees) for PID control
            integral_error: accumulated error over time (degrees * seconds)
            Kp: proportional gain (default: 12)
            Kd: derivative gain (default: 1.5)
            Ki: integral gain (default: 0)
            debug: if True, print debug information
        
        Returns:
            steering_command_us: servo pulse width command (microseconds)
                Typical neutral: 1500 us
                Left: < 1500 us, Right: > 1500 us
        """
        
        theta = obs[0]  # vehicle heading angle (normalized in daemon)
        theta_dot = obs[1]  # vehicle heading rate (derivative of error)
        steering_input_pw = obs[2]  # receiver steering input
        
        # Heading error: difference between current heading and target
        # Wraparound needed: even with normalized angles, difference can exceed ±180
        heading_error = theta - target_theta
        while heading_error > 180:
            heading_error -= 360
        while heading_error < -180:
            heading_error += 360
        
        # Use the servo centerpoint from initialization
        # This handles servos that are mechanically offset from standard 1500 us
        servo_neutral = self.servo_neutral
        steering_deadband_us = self.steering_deadband_us  # +/- deadband around centerpoint
        
        # Steering error: deviation from servo centerpoint (in microseconds)
        steering_delta = steering_input_pw - servo_neutral
        
        # Determine control mode based on whether human is steering
        # If steering input is near neutral, use PID heading hold
        # Otherwise, pass through receiver input (human override)
        if abs(steering_delta) < steering_deadband_us:
            # Heading hold mode: PID based on heading error and thetaprime
            
            # PID terms (in microseconds offset from neutral)
            p_term = Kp * heading_error
            d_term = Kd * theta_dot
            i_term = Ki * integral_error
            
            # Total steering command offset from neutral
            steering_command_offset = p_term + d_term + i_term
            
            # Clamp to reasonable range (e.g., +/- 500 us from neutral)
            steering_command_offset = max(-500.0, min(500.0, steering_command_offset))
            
            # Final servo command
            steering_command_us = servo_neutral + steering_command_offset
            
            if debug:
                print(f"Heading Hold: theta={theta:.2f}°, target={target_theta:.2f}°, error={heading_error:.2f}°, rate={theta_dot:.2f}°/s, "
                      f"PID({p_term:.0f}, {d_term:.0f}, {i_term:.0f}) -> {steering_command_us:.0f}us")
        else:
            # Manual steering active: pass through receiver input
            steering_command_us = steering_input_pw
        
        return steering_command_us
    
    def action(self, steering_command_us):
        """Execute steering action by writing the commanded pulse width to the servo.
        
        Args:
            steering_command_us: steering servo pulse width in microseconds to command
        """
        self._write_servo_pulsewidth(steering_command_us)

    def observe(self):
        """Return observation array: [theta, thetaprime, steering_input_pulse_width, throttle_input_pulse_width].
        
        - theta: vehicle yaw angle (degrees)
        - thetaprime: vehicle yaw rate (degrees/second)
        - steering_input_pulse_width: steering input pulse width from pin 24 (microseconds)
        - throttle_input_pulse_width: throttle input pulse width from pin 23 (microseconds)
        """
        with self._lock:
            steering_pw = self.steering_input_pulse_width
            throttle_pw = self.throttle_input_pulse_width
        return np.array([self.theta, self.thetaprime, steering_pw, throttle_pw])
    
    def log_transition(self, state, action):
        """Log a state-action transition for offline training.
        
        Args:
            state: state array [heading_error, heading_error_dot]
            action: steering servo pulse width command (microseconds) that was executed
        """
        self.data_log.append((state.copy(), action))
    
    def print_mode1_status(self, iteration, obs, heading_error, steering_command_us):
        """Print status for drive mode 1 (PID control).
        
        Args:
            iteration: current iteration number
            obs: observation array [theta, thetaprime, steering_pw, throttle_pw]
            heading_error: current heading error in degrees
            steering_command_us: commanded steering pulse width in microseconds
        """
        print(f"i={iteration:3d} | theta={obs[0]:7.2f}° | target={self.target_theta:7.2f}° | error={heading_error:7.2f}° | "
              f"thetaprime={obs[1]:7.2f}°/s | cmd={steering_command_us:7.0f}us | "
              f"steering_in={obs[2]:7.0f}us | throttle_in={obs[3]:7.0f}us")
    
    def print_mode2_status(self, iteration, obs, steering_command_us, current_time):
        """Print status for drive mode 2 (data collection).
        
        Args:
            iteration: current iteration number
            obs: observation array [theta, thetaprime, steering_pw, throttle_pw]
            steering_command_us: commanded steering pulse width in microseconds (or None if not logging)
            current_time: current time in seconds
        """
        if iteration % 10 != 0:  # Only print every 10th iteration
            return
            
        if self.data_collection_active:
            elapsed = current_time - self.data_collection_start_time
            remaining = self.data_collection_duration - elapsed
            print(f"i={iteration:5d} | [LOGGING] {remaining:.1f}s left | theta={obs[0]:7.2f}° | "
                  f"PID_cmd={steering_command_us:.0f}us | throttle={obs[3]:7.0f}us | "
                  f"reward={reward:7.4f} | samples={len(self.data_log)}")
        else:
            print(f"i={iteration:5d} | [waiting] | theta={obs[0]:7.2f}° | "
                  f"steering={obs[2]:7.0f}us | throttle={obs[3]:7.0f}us")
    
    def print_data_collection_start(self, throttle_pw):
        """Print message when data collection starts."""
        print(f"\n>>> Run {self.run_counter}: Data collection STARTED (throttle={throttle_pw:.0f}us, target={self.target_theta:.2f}°)")
    
    def print_data_collection_complete(self, filename):
        """Print message when data collection completes."""
        print(f">>> Run {self.run_counter}: COMPLETE | Total samples: {len(self.data_log)} saved to {filename}\n")
    
    def print_mode_info(self, mode):
        """Print information about the current drive mode.
        
        Args:
            mode: drive mode number (1, 2, 3, or 4)
        """
        if mode == 1:
            print("=== Drive Mode 1: Pass-Through ===")
            print("Steering input passes directly to servo (no processing).")
            print("Press Ctrl+C to exit.")
            print()
        elif mode == 2:
            print("=== Drive Mode 2: Data Collection (PID Driver) ===")
            print("PID controller drives the vehicle and logs demonstration data.")
            print(f"Data logging will activate when throttle exceeds {int(1500 + self.throttle_threshold)} us")
            print(f"Each run logs for {self.data_collection_duration} seconds")
            print("States: [theta, theta_dot]")
            print("Actions: PID steering commands")
            print("Rewards: computed from heading error and convergence")
            print("All runs saved to single file: pid_demonstrations.npz")
            print("Press Ctrl+C to exit and save all data.")
            print()
        elif mode == 3:
            print("=== Drive Mode 3: Linear Regression Inference ===")
            print("Linear model predicts steering from state: steering = bias + w1*theta + w2*theta_dot")
            print("Direct regression from demonstrations.")
            print("Press Ctrl+C to exit.")
            print()
        elif mode == 4:
            print("=== Drive Mode 4: PID Control ===")
            print("PID heading-hold controller stabilizes vehicle heading.")
            print(f"Data logging will activate when throttle exceeds {int(1500 + self.throttle_threshold)} us")
            print("Press Ctrl+C to exit.")
            print()
    
    def step(self, iteration, dt):
        """Execute one control loop iteration based on the current drive mode.
        
        This method consolidates all control logic for all drive modes.
        
        Args:
            iteration: current iteration number
            dt: time step in seconds
        
        Returns:
            None (handles all control internally)
        """
        
        obs = self.observe()
        steering_input_pw = obs[2]
        throttle_input_pw = obs[3]
        
        # Detect steering input changes for target heading reset
        current_steering_error = steering_input_pw - self.servo_neutral
        previous_steering_error = self.previous_steering_input - self.servo_neutral
        
        # Check if steering returned to neutral (was active, now neutral)
        if abs(previous_steering_error) >= self.steering_deadband_us and abs(current_steering_error) < self.steering_deadband_us:
            self.integral_error = 0.0
            self.reset_target_heading()
        
        # Check if steering direction changed
        elif abs(current_steering_error) >= self.steering_deadband_us and abs(previous_steering_error) >= self.steering_deadband_us:
            if (current_steering_error * previous_steering_error) < 0:
                self.integral_error = 0.0
                self.reset_target_heading()
        
        # Mode-specific control logic
        if self.drive_mode == 1:
            # Pass-through mode
            steering_command_us = steering_input_pw  # Use receiver input directly
            self.action(steering_command_us)
            if iteration % 10 == 0:
                print(f"i={iteration:3d} | theta={obs[0]:7.2f}° | thetaprime={obs[1]:7.2f}°/s | "
                      f"steering_in={obs[2]:7.0f}us | throttle_in={obs[3]:7.0f}us")
        
        elif self.drive_mode == 2:
            # Data collection mode
            if not self.data_collection_active and (throttle_input_pw - self.servo_neutral) > self.throttle_threshold:
                self.data_collection_active = True
                self.data_collection_start_time = iteration * dt
                self.target_theta = obs[0]
                self.integral_error = 0.0
                self.run_counter += 1
                self.print_data_collection_start(throttle_input_pw)
            
            if self.data_collection_active:
                elapsed_time = (iteration * dt) - self.data_collection_start_time
                if elapsed_time >= self.data_collection_duration:
                    self.data_collection_active = False
                    filename = 'demonstrations.npz'
                    self.save_data_log(filename)
                    self.print_data_collection_complete(filename)
            
            if self.data_collection_active:
                steering_command_us = self.PID_action(obs, self.target_theta, self.integral_error)
                heading_error = obs[0] - self.target_theta
                # Normalize heading error to -180 to +180
                while heading_error > 180:
                    heading_error -= 360
                while heading_error < -180:
                    heading_error += 360
                
                self.integral_error += heading_error * dt
                self.integral_error = max(-50.0, min(50.0, self.integral_error))
                
                self.action(steering_command_us)
                
                # State is [heading_error, heading_error_dot] not absolute values
                state = np.array([heading_error, obs[1]])  # error and error_dot (thetaprime)
                action = steering_command_us
                self.log_transition(state, action)
                self.print_mode2_status(iteration, obs, steering_command_us, None, iteration * dt)
            else:
                # Pass-through when not collecting
                steering_command_us = steering_input_pw
                self.action(steering_command_us)
                self.print_mode2_status(iteration, obs, None, None, iteration * dt)
        
        elif self.drive_mode == 3:
            # Linear regression inference mode (using pretrained model)
            if self.linear_agent is None:
                print("ERROR: Linear agent not set! Call vehicle.linear_agent = your_agent before running mode 3")
                return
            
            # Initialize target heading on first iteration
            if iteration == 0:
                self.reset_target_heading()
            
            # Check if human is providing steering input (outside deadband)
            if abs(current_steering_error) >= self.steering_deadband_us:
                # Human override: use human steering input
                steering_command_us = steering_input_pw
                
                if iteration % 10 == 0:
                    print(f"i={iteration:5d} | [HUMAN OVERRIDE] steering={steering_command_us:.0f}us (input)")
            else:
                # Autonomous mode: use linear model
                # Calculate heading error
                heading_error = obs[0] - self.target_theta
                # Normalize heading error to -180 to +180
                while heading_error > 180:
                    heading_error -= 360
                while heading_error < -180:
                    heading_error += 360
                
                # State is [heading_error, heading_error_dot]
                state = np.array([heading_error, obs[1]])  # error and error_dot (thetaprime)
                
                # Predict steering command using linear model
                steering_command_us = self.linear_agent.predict(state)
                
                # Debug: print predictions on first few iterations
                if iteration < 3:
                    print(f"[DEBUG] State: heading_error={state[0]:.2f}°, error_dot={state[1]:.2f}°/s")
                    print(f"[DEBUG] Predicted steering: {steering_command_us:.0f}us")
                    print(f"[DEBUG] Model: bias={self.linear_agent.bias:.2f}, weights={self.linear_agent.weights}")
                
                # Print status periodically
                if iteration % 10 == 0:
                    print(f"i={iteration:5d} | [LINEAR INFERENCE] error={heading_error:7.2f}° | error_dot={obs[1]:6.2f}°/s | "
                          f"steering={steering_command_us:.0f}us | target={self.target_theta:.1f}°")
            
            # Execute the action
            self.action(steering_command_us)
        
        elif self.drive_mode == 4:
            # PID mode
            theta = obs[0]
            heading_error = theta - self.target_theta
            while heading_error > 180:
                heading_error -= 360
            while heading_error < -180:
                heading_error += 360
            self.integral_error += heading_error * dt
            
            steering_command_us = self.PID_action(obs, self.target_theta, self.integral_error)
            self.action(steering_command_us)
            self.print_mode1_status(iteration, obs, heading_error, steering_command_us)
        
        self.previous_steering_input = steering_input_pw
    
    def save_data_log(self, filename='rl_data.npz'):
        """Save collected data to file for offline training.
        
        Saves as numpy arrays: states [heading_error, error_dot], actions
        """
        if len(self.data_log) == 0:
            print("No data to save")
            return
        
        states = np.array([s for s, a in self.data_log])
        actions = np.array([a for s, a in self.data_log])
        
        np.savez(filename, states=states, actions=actions)
        print(f"Saved {len(self.data_log)} transitions to {filename}")
    
    def clear_data_log(self):
        """Clear the data log (e.g., after saving)."""
        self.data_log = []

    def read_input_pulse_width(self, pin):
        """Return the current input pulse width (microseconds) from the background PWM monitor.
        
        Args:
            pin: 23 for throttle, 24 for steering
        
        Returns the pulse width value in microseconds or None if the pin is not monitored.
        Typical servo range: 1000-2000 us.
        """
        if pin == 23:
            return self.throttle_input_pulse_width
        elif pin == 24:
            return self.steering_input_pulse_width
        return None

    def start_pwm_monitor(self, pin, smoothing=0.0):
        """Start a PWM monitor on `pin` using pigpio callbacks.
        """
        if self._pi is None:
            return False
        
        if pin in self._pwm_callbacks:
            return True  # already running
        
        # setup pin as input with pull-down
        try:
            self._pi.set_mode(pin, pigpio.INPUT)
            self._pi.set_pull_up_down(pin, pigpio.PUD_DOWN)
        except Exception as e:
            print(f"start_pwm_monitor on pin {pin}: setup failed: {e}")
            return False
        
        # callback state (per-pin)
        state = {'t_r': None, 't_f': None, 'last_width': None}
        
        def _cb(gpio, level, tick):
            # level: 1=rising, 0=falling, 2=watchdog
            if level == 1:
                # rising edge
                if state['t_r'] is None:
                    state['t_r'] = tick
                else:
                    # compute pulse width (high time in microseconds) from previous cycle if complete
                    if state['t_r'] is not None and state['t_f'] is not None:
                        pulse_width_us = (state['t_f'] - state['t_r']) & 0xFFFFFFFF
                        #Range is 1000-2000 us
                        if pulse_width_us >= 1000 and pulse_width_us <= 2000:
                            if smoothing <= 0:
                                computed_width = pulse_width_us
                            else:
                                if state['last_width'] is None:
                                    computed_width = pulse_width_us
                                else:
                                    computed_width = smoothing * state['last_width'] + (1.0 - smoothing) * pulse_width_us
                            state['last_width'] = computed_width
                            # store in appropriate attribute
                            with self._lock:
                                if pin == 23:
                                    self.throttle_input_pulse_width = computed_width
                                elif pin == 24:
                                    self.steering_input_pulse_width = computed_width
                                elif pin == 27:
                                    self.modeswitch_input_pulse_width = computed_width
                    # start new cycle
                    state['t_r'] = tick
                    state['t_f'] = None
            elif level == 0:
                # falling edge
                state['t_f'] = tick
        
        try:
            cb_handle = self._pi.callback(pin, pigpio.EITHER_EDGE, _cb)
            self._pwm_callbacks[pin] = cb_handle
            print(f"start_pwm_monitor on pin {pin}: callback registered")
            return True
        except Exception as e:
            print(f"start_pwm_monitor on pin {pin}: callback registration failed: {e}")
            return False

    def shutdown(self):
        """Shutdown the vehicle, stop PWM and pigpio connection."""
        if self.drive_mode == 1:
            print("Pass-through mode terminated")
        
        elif self.drive_mode == 2:
            print(f"Total runs completed: {self.run_counter}")
            
            # Save all accumulated data
            if len(self.data_log) > 0:
                filename = 'demonstrations.npz'
                self.save_data_log(filename)
                print(f"All demonstration data saved to {filename} ({len(self.data_log)} transitions)")
        
        elif self.drive_mode == 3:
            print("Linear inference mode terminated")
        
        elif self.drive_mode == 4:
            print("PID control mode terminated")

        try:
            self.pwm.stop()
        except Exception:
            pass

        # close pigpio connection
        if self._pi is not None:
            try:
                self._pi.stop()
            except Exception:
                pass

        print("shutting down")