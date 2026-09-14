from gpiozero import MotionSensor

pir = MotionSensor(17)

print("Waiting for motion...")

while True:
    pir.wait_for_motion()
    print("Motion detected!")

    pir.wait_for_no_motion()
    print("Motion ended")