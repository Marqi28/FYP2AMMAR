import time
import smbus2
from RPLCD.i2c import CharLCD

# Set the correct I2C address
I2C_ADDR = 0x27  # Your LCD is detected at 0x27

# Initialize LCD
lcd = CharLCD(i2c_expander='PCF8574', address=I2C_ADDR, port=1, cols=16, rows=2, charmap='A00')

# Display a test message
lcd.clear()
lcd.write_string("Hello, World!")
time.sleep(2)

lcd.clear()
lcd.write_string("LCD is Working!")

# Keep text displayed for 5 seconds
time.sleep(5)
lcd.clear()
