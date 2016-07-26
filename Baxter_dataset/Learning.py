import numpy as np
from read_data import *
from qsrs import *



def main():
    R = read_data()
    Q = qsr()
    R.read_all_frames()
    # print R.r_data[1]
    # print R.o_data[1]
    Q.set_qsrs(['mos'])
    Q.make_world_trace(R.r_data,R.o_data)
    Q.test()
    # R.update_scene(R.scene+1)

if __name__=='__main__':
    main()
