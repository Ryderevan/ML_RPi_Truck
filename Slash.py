#Import classes, vehicle class is for all vehicle driving functions, time is for loop timing
from time import sleep
import Vehicle

#Set input pins
throttle_pin = 23
steering_pin = 24
servo_pin = 18

#initiatlize vehicle object
Slash = Vehicle.vehicle(throttle_pin, steering_pin, servo_pin)

# Change this to switch modes, 1 = pass-through, 2 = collect data, 3 = linear model inference, 4 = PID control
DRIVE_MODE = 1    
Slash.set_drive_mode(DRIVE_MODE)

#Set PID parameters, these can be tuned for better performance
Slash.Kp = 14
Slash.Kd = 1.5
Slash.Ki = .5

# Initialize control loop parameters
loop_rate = 50.0  # Hz
dt = 1.0 / loop_rate  # seconds per iteration
i = 0

#Main drive loop
try:
    while True:
        # Execute one control step  
        Slash.step(i, dt)
        
        # Increment iteration counter
        i += 1
        sleep(dt)


#Break loop if Ctrl+C is pressed
except KeyboardInterrupt:
    Slash.shutdown()
