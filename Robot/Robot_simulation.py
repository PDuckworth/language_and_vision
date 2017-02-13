from visual import *
import numpy as np
from robot_functions import *


R = Robot()

save = 0
save = 'save'		#saving image

for scene in range(0,1):
    R._initilize_values()
    print 'simulating scene number :',scene
    R.scene = scene
    R._print_scentenses(scene)                  # print the sentences on terminal
    R._generate_scene()                       # place the robot and objects in the initial scene position without saving or motion
    R._move_robot(save)                   # move the robot arm and can save motion as well
    R._save_motion()                       # save the motion into a text file
    #R.__saveSnapshot2()

    R._clear_scene()                            # remove the objects from the scene once it's done
