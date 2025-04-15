import RPi.GPIO as GPIO
import time

buzzer_pin = 18  # GPIO pin where the buzzer is connected

GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
GPIO.setup(buzzer_pin, GPIO.OUT)

print("Buzzer ON")
GPIO.output(buzzer_pin, GPIO.HIGH)  # Turn buzzer ON
time.sleep(2)  # Keep it ON for 2 seconds

print("Buzzer OFF")
GPIO.output(buzzer_pin, GPIO.LOW)  # Turn buzzer OFF

GPIO.cleanup()  # Reset GPIO settings
