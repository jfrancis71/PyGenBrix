#Server code running on Raspberry Pi with Dexter industries BrickPi3, OS=ev3dev
#Controlled by PC client over rpyc

from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_D, OUTPUT_B
import ev3dev2.sensor
import ev3dev2.port
import ev3dev2.sensor.lego
import time

# info from
# https://github.com/ev3dev/ev3dev-lang-python-demo/blob/stretch/platform/brickpi3-motor-and-sensor.py

motora = LargeMotor(OUTPUT_A)
motord = LargeMotor(OUTPUT_D)
motorb = LargeMotor(OUTPUT_B)
wheel_status = -1
p3 = ev3dev2.port.LegoPort(ev3dev2.sensor.INPUT_3)
p3.mode = 'ev3-uart'
p3.set_device = 'lego-ev3-color'
ls = ev3dev2.sensor.lego.ColorSensor( ev3dev2.sensor.INPUT_3 )

def tank_stop():
    global motora
    global motorb
    motora.off( brake = False )
    motord.off( brake = False )

def tank( speed, wheel ):
    global motora
    global motorb
    global motord
    global wheel_status

    col = ls.color
    print( " Start tank, wheel =  ", wheel, " speed = ", speed, flush=True )
    if ( wheel == 1 ):
        motorb.run_to_abs_pos( position_sp = 400, speed_sp = 1020, stop_action = "hold" )
    else:
        motorb.run_to_abs_pos( position_sp = 0, speed_sp = 1020, stop_action = "hold" )

    a_start = motora.position
    a_start_sp = motora.position_sp
    motora.on( -9*speed )
    a_start_sp = motora.position_sp
    motord.on( -4.5 - 9.0*speed )

    return col
