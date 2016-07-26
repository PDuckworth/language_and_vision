#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace
import argparse

def pretty_print_world_qsr_trace(which_qsr, qsrlib_response_message):
    print(which_qsr, "request was made at ", str(qsrlib_response_message.req_made_at)
          + " and received at " + str(qsrlib_response_message.req_received_at)
          + " and finished at " + str(qsrlib_response_message.req_finished_at))
    print("---")
    print("Response is:")
    for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
        foo = str(t) + ": "
        for k, v in zip(qsrlib_response_message.qsrs.trace[t].qsrs.keys(),
                        qsrlib_response_message.qsrs.trace[t].qsrs.values()):
            foo += str(k) + ":" + str(v.qsr) + "; "
        print(foo)

class qsr():
    """docstring for qsr"""
    def __init__(self):
        # ****************************************************************************************************
        # create a QSRlib object if there isn't one already
        self.qsrlib = QSRlib()

    def set_qsrs(self, qsrs_types):
        # ****************************************************************************************************
        # parse command line arguments
        options = sorted(self.qsrlib.qsrs_registry.keys())
        # print(options)
        self.which_qsr = []
        for q in qsrs_types:
            if q in options:
                self.which_qsr.append(q)
            else:
                raise ValueError("qsr not found, keywords: %s" % options)
        # print self.which_qsr

    def make_world_trace(self,robot,objects):
        # ****************************************************************************************************
        # make some input data
        self.world = World_Trace()
        ob_states = {}
        # adding robot states
        t = 1
        for frame in robot:
            if frame%1==0:
                # Right arm
                x1 = robot[frame]['Right_Gripper']['R_x']
                y1 = robot[frame]['Right_Gripper']['R_y']
                z1 = robot[frame]['Right_Gripper']['R_z']+.04
                g1 = robot[frame]['Right_Gripper']['R_gripper']
                if 'Right_Gripper' not in ob_states.keys():
                    ob_states['Right_Gripper'] = [Object_State(name='Right_Gripper', timestamp=t, x=x1,y=y1,z=z1)]
                else:
                    ob_states['Right_Gripper'].append(Object_State(name='Right_Gripper', timestamp=t, x=x1,y=y1,z=z1))
                # Left arm
                x = robot[frame]['Left_Gripper']['L_x']
                y = robot[frame]['Left_Gripper']['L_y']
                z = robot[frame]['Left_Gripper']['L_z']+.04
                if 'Left_Gripper' not in ob_states.keys():
                    ob_states['Left_Gripper'] = [Object_State(name='Left_Gripper', timestamp=t, x=x, y=y, z=z)]
                else:
                    ob_states['Left_Gripper'].append(Object_State(name='Left_Gripper', timestamp=t, x=x, y=y, z=z))
                # adding objects states
                for o in objects[frame]:
                    obj = 'obj_'+str(o)
                    # if g1 < 70:
                    x2 = objects[frame][o]['X']
                    y2 = objects[frame][o]['Y']
                    z2 = objects[frame][o]['Z']
                    # if g1>=70:
                    #     x2 = objects[1][o]['X']
                    #     y2 = objects[1][o]['Y']
                    #     z2 = objects[1][o]['Z']

                    if obj not in ob_states.keys():
                        ob_states[obj] = [Object_State(name=obj, timestamp=t, x=x2,y=y2,z=z2)]
                    else:
                        ob_states[obj].append(Object_State(name=obj, timestamp=t, x=x2,y=y2,z=z2))

                # x1 = int(robot[frame]['Right_Gripper']['R_x']*100)
                # y1 = int(robot[frame]['Right_Gripper']['R_y']*100)
                # z1 = int((robot[frame]['Right_Gripper']['R_z']+.04)*100)
                #
                # x2 = int(objects[frame][o]['X']*100)
                # y2 = int(objects[frame][o]['Y']*100)
                # z2 = int(objects[frame][o]['Z']*100)
                # print t,frame,x1,x2,'....',y1,y2,'....',z1,z2,'....',g1
                t+=1

        # for obj in ob_states:
        #     self.world.add_object_state_series(ob_states[obj])
        self.world.add_object_state_series(ob_states['Right_Gripper'])
        ob_states = {}



    def test(self):
        # ****************************************************************************************************
        # dynammic_args = {'argd': {"qsr_relations_and_values" : {"Touch": 0.5, "Near": 6, "Far": 10}}}
        # make a QSRlib request message
        # dynammic_args = {"qtcbs": {"no_collapse": True, "quantisation_factor":.001, "validate":False, "qsrs_for":[("Right_Gripper","obj_0")] }}
        dynammic_args = {"mos": {"quantisation_factor":.0005}}

        #dynammic_args={"tpcc":{"qsrs_for":[("o1","o2","o3")] }}

        qsrlib_request_message = QSRlib_Request_Message(self.which_qsr, self.world, dynammic_args)
        # request your QSRs
        qsrlib_response_message = self.qsrlib.request_qsrs(req_msg=qsrlib_request_message)

        # ****************************************************************************************************
        # print out your QSRs
        pretty_print_world_qsr_trace(self.which_qsr, qsrlib_response_message)
