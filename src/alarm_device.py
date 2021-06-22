from gpiozero import LED
from time import sleep

def start_alarm():
    sound_alarm = LED(17)
    sound_alarm.on()
    visual_alarm = LED(27)
    visual_alarm.on()
    
def stop_alarm():
    sound_alarm = LED(4)
    sound_alarm.on()
    visual_alarm = LED(5)
    visual_alarm.off()
    
if __name__ == "__main__":
    print("Module Rpi Output")
    start_alarm()
    sleep(2)
    stop_alarm()