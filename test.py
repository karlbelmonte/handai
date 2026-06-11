import serial
import time

# Create the connection
# 'COM3' is your port; 9600 is the speed
ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2) # Vital: Wait for Arduino to reboot after connecting

def send_angle(motor_num, angle):
    command = f"{motor_num}:{angle}\n" # Format: "0:180"
    ser.write(command.encode())
    print(f"Sent: {command.strip()}")

# Move Motor 0 to 90 degrees, then Motor 5 to 180 degrees
send_angle(0, 90)
time.sleep(1)
send_angle(5, 180)

ser.close()