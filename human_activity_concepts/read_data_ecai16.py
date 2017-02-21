def get_events(dir, mean_window, conv_net=True, segmented=False):
    """From the skeleton detections, get the data into an event class """

    print "getting events from: %s" % dir

    """GET CONV_NET SK TRACKS"""
    conv_dir = dir.split("/")
    conv_dir[-2] = 'Data_ConvNet_skeleton'
    conv_dir = ("/").join(conv_dir)

    dirs = get_immediate_dir(dir)
    meta_cnt = 0
    accumulate = []

    for d in sorted(dirs):
        d1 = dir+'/'+d+'/'
        # print d1

        if conv_net: d_sk = conv_dir+'/'+d+'/'+'skeleton/'
        else: d_sk = d1+'skeleton/'
        d_robot = d1+'robot/'
        uuid = d.split('_')[1]

        # Get the event labels
        labels = {}
        if os.path.isfile(os.path.join(d1, 'labels.txt')):
            f1 = open(os.path.join(d1, 'labels.txt'), 'r')
            for count, line in enumerate(f1):
                line_ = line.split('\n')[0].split(':')
                activity_class = line_[1]

                if activity_class != 'making_tea':
                    start, end = line_[2].split(',')
                    labels[count] = (activity_class, int(start), int(end))

            # for count, line in enumerate(f1):
            #     line_ = line.split('\n')[0].split(':')
            #     if line_[0] == 'label' and line_[1] != 'making_tea':
            #         start, end = line_[2].split(',')
            #         labels[line_[1]] = (int(start), int(end))

        sk_files = [f for f in os.listdir(d_sk) if os.path.isfile(os.path.join(d_sk, f))]

        if not segmented:
            # filter the first and final 15 frames from the non segmented videos.
            if len(labels) is 0:
                labels = {0: ("NA", 15, len(sk_files)-15)}
            else:
                classes = ":".join([ a for (a, s, e) in labels.values()])
                labels = {0 : (classes, 5, len(sk_files)-15)}

        #     try:
        #         # print labels
        #         start = min([int(s) for (a, s, e) in labels.values() ])
        #         end = max([int(e) for (a, s, e) in labels.values() ])
        #         # print ">", start, end
        #     except ValueError: # If no rows in the labels text file, this will skip the recording
        #         continue
                    # labe`ls = {":".join(labels.keys()): (5, len(sk_files)-15)}

        # Get the robot's region location  - Always the same...
        # if os.path.isfile(os.path.join(d1, 'meta.txt')):
        #     f1 = open(os.path.join(d1, 'meta.txt'), 'r')
        #     for count, line in enumerate(f1):
        #         if count == 0: region_id = line.split('\n')[0].split(':')[1]
        #         elif count == 1: region = line.split('\n')[0].split(':')[1]
        # else:
        #     region="unknown"
        #     region_id = 0
        # try:
        #     something = region
        # except UnboundLocalError:
        #     meta_cnt+=1
        region = "Kitchen"
        region_id = 1
        print "  ", labels

        for cnt, (event_label, start_frame, end_frame) in labels.items():

            e = event(uuid, event_label, d1, start_frame, end_frame, region, region_id)

            for file in sorted(sk_files):
                frame = int(file.split('.')[0].split('_')[1])
                # print file, frame

                if not segmented and (end_frame - start_frame) < 10: continue  # remove any videos which cannot be median filtered properly

                if frame < int(start_frame) or frame > int(end_frame): continue

                # If there are more than 30 frames in the event, use every other frame.
                if int(end_frame) - int(start_frame) > 30:
                    if frame % 2 != 0: continue

                e.skeleton_data[frame] = {}
                e.sorted_timestamps.append(frame)

                f1 = open(d_sk+file,'r')

                for count,line in enumerate(f1):
                    if count == 0:
                        t = line.split(':')[1].split('\n')[0]
                        e.sorted_ros_timestamps.append(np.float64(t))

                    # read the joint name
                    elif (count-1)%10 == 0:
                        j = line.split('\n')[0]
                        e.skeleton_data[frame][j] = []
                    # read the x value
                    elif (count-1)%10 == 2:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.skeleton_data[frame][j].append(a)
                    # read the y value
                    elif (count-1)%10 == 3:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.skeleton_data[frame][j].append(a)
                    # read the z value
                    elif (count-1)%10 == 4:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.skeleton_data[frame][j].append(a)

            # apply filter and create World_Trace
            e.apply_median_filter(mean_window)
            # print "\nunfiltered:", e.skeleton_data[10]
            # print "\nfiltered:", e.filtered_skeleton_data[0]

            # read robot data
            r_files = [f for f in os.listdir(d_robot) if os.path.isfile(os.path.join(d_robot,f))]
            for file in sorted(r_files):
                frame = int(file.split('.')[0].split('_')[1])
                e.robot_data[frame] = [[],[]]
                f1 = open(d_robot+file,'r')
                for count,line in enumerate(f1):
                    # read the x value
                    if count == 1:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.robot_data[frame][0].append(a)
                    # read the y value
                    elif count == 2:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.robot_data[frame][0].append(a)
                    # read the z value
                    elif count == 3:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.robot_data[frame][0].append(a)
                    # read roll pitch yaw
                    elif count == 5:
                        ax = float(line.split('\n')[0].split(':')[1])
                    elif count == 6:
                        ay = float(line.split('\n')[0].split(':')[1])
                    elif count == 7:
                        az = float(line.split('\n')[0].split(':')[1])
                    elif count == 8:
                        aw = float(line.split('\n')[0].split(':')[1])
                        # ax,ay,az,aw
                        roll, pitch, yaw = euler_from_quaternion([ax, ay, az, aw])    #odom
                        pitch = 10*math.pi / 180.   #we pointed the pan tilt 10 degrees
                        e.robot_data[frame][1] = [roll,pitch,yaw]

            # add the map frame data for skeleton
            for frame in e.sorted_timestamps:
                """Note frame does not start from 0. It is the actual file frame number"""

                e.map_frame_data[frame] = {}
                xr, yr, zr = e.robot_data[frame][0]
                yawr = e.robot_data[frame][1][2]
                pr = e.robot_data[frame][1][1]

                # because the Nite tracker has z as depth, height as y and left/right as x
                # we translate this to the map frame with x, y and z as height.
                for joint, (y,z,x,x2d,y2d) in e.filtered_skeleton_data[frame].items():
                    # transformation from camera to map
                    rot_y = np.matrix([[np.cos(pr), 0, np.sin(pr)], [0, 1, 0], [-np.sin(pr), 0, np.cos(pr)]])
                    rot_z = np.matrix([[np.cos(yawr), -np.sin(yawr), 0], [np.sin(yawr), np.cos(yawr), 0], [0, 0, 1]])
                    rot = rot_z*rot_y

                    # robot's position in map frame
                    pos_r = np.matrix([[xr], [yr], [zr+1.66]])

                    # person's position in camera frame
                    pos_p = np.matrix([[x], [-y], [z]])

                    # person's position in map frame
                    map_pos = rot*pos_p+pos_r
                    x_mf = map_pos[0][0]
                    y_mf = map_pos[1][0]
                    z_mf = map_pos[2][0]

                    j = (x_mf, y_mf, z_mf)
                    e.map_frame_data[frame][joint] = j

            e.get_world_frame_trace(get_soma_objects(e.region))
            if len(e.sorted_timestamps) >= 5:
                if segmented:
                    save_event(e, "Events/Events_Seg")
                else:
                    save_event(e, "Events/Events_noSeg")
                accumulate.append((end_frame-start_frame))
            else:
                print "  >dont save me - recording too short."
    return accumulate
