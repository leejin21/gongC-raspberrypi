from gpiozero import LED
from time import sleep

red_led = LED(17)
green_led = LED(27)

while True:
    red_led.on()
    green_led.off()

    sleep(2)

    red_led.off()
    green_led.on()

    sleep(2)