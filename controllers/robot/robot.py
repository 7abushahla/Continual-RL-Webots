import numpy as np
from deepbots.robots import CSVRobot


def normalize_to_range(value, min, max, new_min, new_max):
    value = float(value)
    min = float(min)
    max = float(max)
    new_min = float(new_min)
    new_max = float(new_max)
    return (new_max - new_min) / (max - min) * (value - max) + new_max


class UnifiedEpuckRobot(CSVRobot):
    def __init__(self):
        super().__init__()
        self.setup_sensors()
        self.setup_motors()
        self.motor_speeds = [0.0, 0.0]

        # ── DEBUG: print the emitter/receiver channels ──
        print(f"[Robot] Emitter channel = {self.emitter.getChannel()}, "
              f"Receiver channel = {self.receiver.getChannel()}")


    def setup_sensors(self):
        """Setup all sensors (8 proximity + 3 ground)"""
        # Proximity sensors
        self.proximity_sensors = []
        for i in range(8):
            sensor = self.getDevice(f'ps{i}')
            sensor.enable(self.timestep)
            self.proximity_sensors.append(sensor)
            
        # Ground sensors
        self.ground_sensors = []
        for i in range(3):
            sensor = self.getDevice(f'gs{i}')
            sensor.enable(self.timestep)
            self.ground_sensors.append(sensor)

    def create_message(self):
        """Create message with all 11 sensor readings"""
        message = []
        
        # Add 8 proximity sensor readings
        for sensor in self.proximity_sensors:
            message.append(sensor.getValue())
            
        # Add 3 ground sensor readings
        for sensor in self.ground_sensors:
            message.append(sensor.getValue())
            
        return message

    def use_message_data(self, message):
        """Handle motor commands from supervisor"""
        if len(message) != 2:
            return
            
        # Extract gas and wheel commands
        wheel = float(message[0])  # Wheel command
        gas = float(message[1])  # Gas command
        
        # Mapping gas from [-1, 1] to [0, 6] to use full speed range
        gas = (gas + 1) * 3  # Changed from *2 to *3 to get [0, 6]
        gas = np.clip(gas, 0, 6.0)  # Changed from 4.0 to 6.0
        
        # Mapping turning rate from [-1, 1] to [-3, 3] for sharper turns
        wheel *= 3  # Changed from *2 to *3
        wheel = np.clip(wheel, -3, 3)  # Changed from -2,2 to -3,3
        
        # Apply gas to both motor speeds, add turning rate to one, subtract from other
        self.motor_speeds[0] = gas + wheel
        self.motor_speeds[1] = gas - wheel
        
        # Clip final motor speeds to [0, 6.28] to match MAX_SPEED
        self.motor_speeds = np.clip(self.motor_speeds, 0, 6.28)
        
        # Apply motor speeds
        self._set_velocity(self.motor_speeds[0], self.motor_speeds[1])

    def setup_motors(self):
        """Setup motor devices"""
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self._set_velocity(0.0, 0.0)
    
    def _set_velocity(self, v_left, v_right):
        """Set wheel velocities"""
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(v_left)
        self.right_motor.setVelocity(v_right)

if __name__ == '__main__':
    # Create and run the robot controller
    robot_controller = UnifiedEpuckRobot()
    robot_controller.run() 