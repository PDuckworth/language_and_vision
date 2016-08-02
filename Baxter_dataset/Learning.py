import numpy as np
from read_data import *
from qsrs import *



def main():
    R = read_data()
    Q = qsr()
    R.read_all_frames()
    R.apply_filter()

    Q.set_qsrs(['mos'])
    Q.make_world_trace(R.r_data,R.o_data)
    Q.motion_qsr()
    # R.update_scene(R.scene+1)

if __name__=='__main__':
    main()
