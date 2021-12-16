from gpiozero import LED
from time import sleep

red_led = LED(17)
green_led = LED(27)

def makeRedLEDOn():
    red_led.on()
    green_led.off()

def makeGreenLEDOn():
    red_led.off()
    green_led.on()
