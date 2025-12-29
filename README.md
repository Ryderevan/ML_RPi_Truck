# ML_RPi_Truck
Training an ML model to control a radio control truck with custom raspberrypi based hardware. The truck used as a demo is a 1:10 scale Losi Baja Rey, shown here: 
<img width="1600" height="1600" alt="image" src="https://github.com/user-attachments/assets/b98e383b-53ef-438e-bb69-ec668b7c850b" />


## Task

The proposed task is to keep the vehicle in a straight line. When the vehicle is perturbed by bumps in terrain or other inputs, it will most likely not end up driving in a straight line if no driver input is given. If we define the angle of the vehicle as its heading, and we define a target heading as an arbitrary direction, we can calculate an angle theta, which is the difference between the two angles. This can also be defined as the vehicles error. 

![VehicleDrawing](https://github.com/user-attachments/assets/aee1eedb-06cf-4c99-942b-2e0dc97f50e6)



## Control System
A traditional control system would sense the vehicle heading, calculate the error given the target, and then adjust the steering. It would do this repeatedly, ideally reducing the error to 0. The traditional control system of choice for this problem would likely be a PID controller, which has three components, Proportion(P), Integral(I), and Derivative(D). This can be implemented where the control input is calculated by:

	u = Kp*Error + Ki*Error*t + Kd*Error_prime

Kp, Ki, and Kd are the controllers coefficients, while each corresponding term is a calculation based on the vehicles error. 
This activity will not be based on a PID controller however, it will be based on a linear regression model, a type of supervised machine learning.
Rather than define these coefficients, we will collect data from an 'expert', for this example, a student driving the vehicle themselves. The students would do 10 to 15 test runs where we collect data on the vehicle state and actions taken by the expert. Given these state-action pairs, we train a regression model to learn the relationships between the states and actions. Our output of the model will be two coefficients, effectively Kp and Kd in the PID example. We then import that model on the vehicle, and test its performance in a real life driving example.


## The Activity
The topics and an estimated timeline is shown here, it should take about 4 hours to complete this activity:

	1. Intro to problem (10)
	
	2. Background knowledge required(30)
		a. Pwm
			i. Demo of pulse width changing on oscilloscope with steering moving
		b. Multithreading, daemons
			i. Quick thought experiment of why multithreading is important(how do do things around the house, would you stand by the dishwasher waiting for it to finish? No, you would go do something else)
		c. Linear regression
			i. Concept-given a x value, predict y.
			ii. Plot points and draw a line
			
	3. Go outside, do a demo of the goal solution, problem with vehicle sliding out (15 mins)
	
	4. Handout vehicles, introduce concept of teams(10)
		a. Hardware(electrical)
		b. Software
		c. Data scientist
		d. Driver
		
	5. Hardware and wiring (30)
		a. Intro to task of wiring
		b. Show diagram, walk through devices
		c. Team connects wires
		d. Check wires
		
	6. Software(45)
		a. Intro to task of writing the software
		b. Show software flow
		c. Team writes software
		d. Check software 
		
	7. Plug in batteries(10)
		a. Check operation
		
	8. Driving instruction, students drive briefly(25)
		a. Choose best drivers for training data
		
	9. Training(30)
		a. Put vehicles in training mode
		b. Collect data
		c. Review results
		d. Train model, review errors
		e. Would be nice to have a few bad models
		
	10. Inference(30)
		a. Put vehicles in Inference mode
		
	11. Test outside


## Hardware

These hobby grade radio control vehicles are built in a modular fashion, and the signals being passed between the components can be intercepted and modified with a customer controller. A standard vehicle is made up of a receiver, a servo, an electronic speed controller (ESC), and a motor. The receiver takes input from the transmitter, then sends the signals to the servo to control steering, and the ESC to control the motor. In this project, receiver servo signal output is being input to the raspberry pi and overriding the servo control. The receivers ESC output is being monitored by the raspberry pi, but will not be used to override the signal being sent to the ESC.

The Raspberry Pi Zero W 2 will be used here and will be referred to as the 'controller' for the rest of this documentation. With its ability to output Pulse Width Modulation (PWM), its small size, and its ability to connect to a remote computer via wifi for monitoring, it works well in this scenario. To get the vehicles angle, we will use a LSM6DSOX IMU. The gyro within the IMU provides us with a rate of change of angle, which we can then use dead reckoning to get the angle by integrating  the rate of change over time. 
<img width="2000" height="980" alt="image" src="https://github.com/user-attachments/assets/d98d5a27-f34d-4b4c-bb1c-ec86a1b1103e" />

Wiring diagram: 
![BajaReyDiagrams-Wiring](https://github.com/user-attachments/assets/38872672-c637-428c-9f95-a17ad9e93b63)


## Data and ML

After collecting data from an expert driver, we can train a regression model and test our results. For this scenario, we can use two features, error, and error_rate. These correspond to the P and D components of the PID controller, as mentioned above. A two feature model will be of the general form:

	u = m1x1 + m2x2 + bias

In our problem, we can expand to:

	u = m1*error + m2*error_rate + bias

After training our model to learn m1, m2, and bias, we can compare the predicted values to the actual values, as shown here: 

<img width="1389" height="990" alt="image" src="https://github.com/user-attachments/assets/3c2aa6df-e12f-461b-9a99-f19bbf6ac8b5" />


## Testing on a vehicle

Once a model has been trained, we can then change vehicle modes to use the model to control the steering. Inference mode will use the model to infer the steering given the vehicle state, or will allow the user to override if steering is provided. This allowes the vehicle to be driven normally, but will let the ML model take over if no steering input is given.
