# Start RPYC and image web browser on Pi before running

import argparse
import PyGenBrix.utils.dataset_utils as ds
import rpyc

ap = argparse.ArgumentParser(description="gather_data")
ap.add_argument("--ip_address") # ip address of Pi
ap.add_argument("--folder", default="data")
ap.add_argument("--rand", action="store_true")  # Use Rand for random movement between frames

conn = rpyc.classic.connect(ns.ip_address)
conn.execute("from ev3dev2.motor import LargeMotor, OUTPUT_A")
conn.execute("m = LargeMotor(OUTPUT_A)")

seq = 0
degrees = 2*360
while True:
    image = ds.import_http_image("http://"+ns.ip_address+":8080?action=snapshot")
    image.save(ns.folder+"/file_"+str(seq)+".jpg")
    if ns.rand:
        degrees = random.randint(-1080,+1080)
    conn.execute("m.on_for_degrees(75, " + str(degrees) + ")")
    seq += 1
