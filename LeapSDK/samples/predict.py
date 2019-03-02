import os, sys, inspect, thread, time
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = os.path.abspath(os.path.join(src_dir, '../lib'))
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap, sys, thread, time
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

class LeapModel():

    def __init__(self, filename):
        self.clf = pickle.load(open(filename, 'rb'))

    def predict_single(self, features):
        return self.clf.predict([features]), self.clf.predict_proba([features])

keep_program = True
clf = LeapModel('model_final.sav')

class SampleListener(Leap.Listener):
    counter = 0
    feature = []
    history = []

    def on_init(self, controller):
        self.controller = controller
        print "Move in your hand to start..."

    def on_connect(self, controller):
        print "Sensor Connected"

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected"

    def on_exit(self, controller):
        pass

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()
        # print "Frame id: %d, timestamp: %d, hands: %d, fingers: %d, tools: %d, gestures: %d" % (
            # frame.id, frame.timestamp, len(frame.hands), len(frame.fingers), len(frame.tools), len(frame.gestures()))

        # Get hands
        self.counter += 1
        if self.counter % 60 != 0: return
        for hand in frame.hands:
            self.feature = []

            # Get the hand's normal vector and direction
            normal = hand.palm_normal
            direction = hand.direction

            # Calculate the hand's pitch, roll, and yaw angles
            self.feature.extend(
                            [direction.pitch * Leap.RAD_TO_DEG,
                            normal.roll * Leap.RAD_TO_DEG,
                            direction.yaw * Leap.RAD_TO_DEG])

            # Get fingers
            for finger in hand.fingers:

                # Get bones
                for b in range(0, 4):
                    bone = finger.bone(b)
                    # print "      Bone: %s, start: %s, end: %s, direction: %s" % (
                    #     self.bone_names[bone.type],
                    #     bone.prev_joint,
                    #     bone.next_joint,
                    #     bone.direction)
                    self.feature.extend(
                        [bone.direction[0],
                        bone.direction[1],
                        bone.direction[2]]
                    )

            # print(len(self.feature))
            # time.sleep(2.5)
            label, proba = clf.predict_single(self.feature)
            print 'Sign Meaning: ', label
            if len(self.history) == 3:
                del self.history[0]
                self.history.append(label)
            else:
                self.history.append(label)


            if len(self.history) == 3 and label == "stop":
                for index in range (0, 3):
                    if self.history[index] != "stop":
                        return
                    if index == (len(self.history) - 1):
                        controller.remove_listener(self)
                        print 'Program existed'


    def state_string(self, state):
        if state == Leap.Gesture.STATE_START:
            return "STATE_START"

        if state == Leap.Gesture.STATE_UPDATE:
            return "STATE_UPDATE"

        if state == Leap.Gesture.STATE_STOP:
            return "STATE_STOP"

        if state == Leap.Gesture.STATE_INVALID:
            return "STATE_INVALID"


def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)
        sys.exit(0)



if __name__ == "__main__":
    main()
