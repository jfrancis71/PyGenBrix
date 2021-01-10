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

def tank( speed, wheel ):
    global motora
    global motorb
    global motord
    global wheel_status

    print( " Start tank, wheel =  ", wheel, " speed = ", speed, flush=True )
    print( "Wheel ", wheel, " wheel_status ", wheel_status )
    if ( wheel != wheel_status ):
        if ( wheel == 1 ):
            print( "Starting wheel right", flush = True )
            motorb.run_to_rel_pos( position_sp = 400, speed_sp = 400, stop_action = "hold" )
        else:
            print( "Starting wheel left", flush = True )
            motorb.run_to_rel_pos( position_sp = -400, speed_sp = 400, stop_action = "hold" )

    print( "Completed wheel start", flush = True )
    a_start = motora.position
    a_start_sp = motora.position_sp
    print( " Start  A current_pos = ", a_start, "sp=", a_start_sp, flush=True )
    motora.run_to_rel_pos( position_sp = -90*speed, speed_sp = 100, stop_action = "hold" )
    a_start_sp = motora.position_sp
    print( " Finished  A Start", a_start_sp, flush=True )
    print( " Start  D", flush=True )
    motord.run_to_rel_pos( position_sp = -45 - 90*speed, speed_sp = 100, stop_action = "hold" )
    print( " Finished  D Start", flush=True )
    print( "Waiting for A to finish ", flush=True)
    motora.wait_while( "running" )
    print( "A Finished ", flush=True)
    print( "Waiting for D to finish ", flush=True)
    motord.wait_while( "running" )
    print( "D Finished ", flush=True)
    print( "Waiting for wheel to finish", flush = True )
    motorb.wait_while( "running" )
    print( "Wheel finished" )
    wheel_status = wheel

    print( "Getting color", flush=True )
    col = ls.color

    print( "Finished color", flush = True )
    print( "Finshed Tank", flush=True )
    return col
