# Start RPYC and image web browser on Pi before running

import argparse
import random
import PyGenBrix.utils.dataset_utils as ds
import rpyc
import json

ap = argparse.ArgumentParser(description="gather_data")
ap.add_argument("--ip_address") # ip address of Pi
ap.add_argument("--folder", default="data")
ap.add_argument("--process")  # Use Rand for random movement between frames

ns = ap.parse_args()

conn = rpyc.classic.connect(ns.ip_address)
conn.execute("from ev3dev2.motor import LargeMotor, OUTPUT_A")
conn.execute("m = LargeMotor(OUTPUT_A)")

seq = 0
actions = []
base_degrees = 2*360
degrees = 0.0
while True:
    image = ds.import_http_image("http://"+ns.ip_address+":8080?action=snapshot")
    image.save(ns.folder+"/file_"+str(seq)+".jpg")
    if ns.process == "rand":
        degrees = random.randint(-1080,+1080)
    elif ns.process == "forward":
        degrees = base_degrees
    elif ns.process == "action":
        action = random.randint(0,2)
        if action == 0:
            degrees = 0.0
        elif action == 1:
            degrees = base_degrees
        elif action == 2:
            degrees = -base_degrees
        actions.append(action)
        with open(ns.folder+"/action.json", 'w') as f:
            json.dump(actions, f)
    conn.execute("m.on_for_degrees(75, " + str(degrees) + ")")
    seq += 1
