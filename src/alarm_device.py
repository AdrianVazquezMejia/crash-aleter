from gpiozero import LED
from time import sleep

def alarm_device(pedestrian_flag, car_flag, pedestrian_id, car_id):
    if pedestrian_flag and car_flag:
        print(f'ALARM!!! ALARM!!! Person{pedestrian_id} is going to collide with Car{car_id}')
        print("Module Rpi Output")
        sound_alarm = LED(17)
        sound_alarm.on()
        visual_alarm = LED(27)
        visual_alarm.on()      
        sleep(2)
        sound_alarm = LED(4)
        sound_alarm.off()
        visual_alarm = LED(5)
        visual_alarm.off()
    else:
        print("No cause for alarm :))
    return


#def start_alarm():
#    sound_alarm = LED(17)
#    sound_alarm.on()
#    visual_alarm = LED(27)
#    visual_alarm.on()
    
#def stop_alarm():
#    sound_alarm = LED(4)
#    sound_alarm.on()
#    visual_alarm = LED(5)
#    visual_alarm.off()
    
#if __name__ == "__main__":
#    print("Module Rpi Output")
#    start_alarm()
#    sleep(2)
#    stop_alarm()
