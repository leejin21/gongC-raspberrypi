import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(17,GPIO.OUT)
GPIO.setup(27,GPIO.OUT)

while(True):
    GPIO.output(17,False)
    GPIO.output(27,True)
    time.sleep(2)
    GPIO.output(17,True)
    GPIO.output(27,False)
    time.sleep(2)