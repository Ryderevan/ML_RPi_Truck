#Import classes, vehicle class is for all vehicle driving functions, time is for loop timing
from time import sleep
import Vehicle

#Set input pins
throttle_pin = 23
steering_pin = 24
servo_pin = 18

#initiatlize vehicle object
Slash = Vehicle.vehicle(throttle_pin, steering_pin, servo_pin)

# Change this to switch modes, Mode 1 for pass through, Mode 2 to collect data and mode 3 to run linear model inference
DRIVE_MODE = 1    
Slash.set_drive_mode(DRIVE_MODE)

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
