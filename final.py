import sys
import numpy as np
from copy import deepcopy
from math import pi
from lib.calculateFK import FK
from lib.solveIK import IK
import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds


def block_pos( pose, q):
    # takes in cube pos wrt to cam frame and current robot sate q (1,7)
    #outputs cube pos in world frame
    H_cam= np.array([[ 0.,   -1.,    0.,    0.05],
       [ 1.  ,  0.,    0.,    0.  ],
       [ 0. ,   0. ,   1.,   -0.06],
       [ 0.  ,  0. ,   0.   , 1.  ]])

    fk=FK()
    T0e=fk.forward(q)[1]
    bp=T0e @ H_cam @ pose
    #print (bp)
    return bp  #returns 4x4 matrix of the block

### scan and grab ##

def scan():    #scans cube pos return array of seen cube pos
    
    pos_array= []
    for (name, pose) in detector.get_detections():
        print(name,'\n',pose)
        position = block_pos(pose, q) 
        print (position)
        pos_array.append(position)
    for i in pos_array:       
        i[0:3,2] = np.array([0,0,-1])
        i[3,0:3] = np.array([0,0,0])
        i[3,3]=1
        i[1,3]=i[1,3]-0.09  #offset 
        i[0,0]=  0    #0.70710678118
       # s= (1-c**2)**0.5
        i[1,0]= -1    #-0.70710678118
        i[0,1]= -1   #0.70710678118
        i[1,1]= 0    #0.70710678118
        i[2,0:2]=0
    return pos_array
## places block and releases ##

def place_block(q_target, q_mid):        # places static block avoiding collision with other SB
    #for cube in range(len(pos_array))::
    
    arm.safe_move_to_position(q_mid) 

    #to place the block on the target square
    arm.safe_move_to_position(q_target)  # release block on target
    arm.exec_gripper_cmd(0.10, 20)
    #arm.safe_move_to_position(start_position)

def dynamic_blue(q):
   # q is the state of the robot at the target stacking pos.
    dynamic_sweep_end_position = np.array([-pi/1.8, 1.2,  0.2, -0.65, -2.89, 0.99*pi, -0.9*pi])
    dynamic_sweep_start_position = np.array([-pi/1.8, 0.32,  0.2, -2.17, -2.89, 0.77*pi, -pi+pi/5.2])
    dynamic_start_position = np.array([-pi/2-pi/9, 0.5,  0.02, -1, 0.1, 0.35+pi/2, -0.25*pi])

    counter=0
    arm.safe_move_to_position(dynamic_start_position)  # dynamic home
    arm.exec_gripper_cmd(0.1,50)     #open gripper
    arm.safe_move_to_position(dynamic_sweep_start_position) # starts sweeping blocks from here
    arm.safe_move_to_position(dynamic_sweep_end_position)   # end sweep from here
    arm.exec_gripper_cmd(0.04, 30)   
    gripper=arm.get_gripper_state()
    grip_dist=gripper["position"][0]+gripper["position"][1]
   
    if grip_dist < 0.025:                                   #check if block is grabbed
        arm.safe_move_to_position(dynamic_start_position)   #if not go back to start
    else:
        arm.safe_move_to_position(dynamic_start_position)   #if grabbed, move here 
        arm.safe_move_to_position(q)                        #move to stacking pos
        arm.exec_gripper_cmd(0.1, 30)                       #release block
        counter=1  ## if block is grabbed increment counter by 1

    return counter

def dynamic_red(q):
    dynamic_sweep_end_position = np.array([pi/2, 1.2,  0.2, -0.65, -2.89   , 0.99*pi, -0.9*pi])
    dynamic_sweep_start_position = np.array([pi/2-pi/20, 0.32,  0.2, -2.17, -2.89, 0.77*pi, -pi+pi/5.2])
    dynamic_start_position = np.array([pi/2-pi/20, 0.5,  0.02, -1, 0.1, 0.35+pi/2, -0.25*pi])

    counter=0
    arm.safe_move_to_position(dynamic_start_position)  # dynamic home
    arm.exec_gripper_cmd(0.1,50)     #open gripper
    arm.safe_move_to_position(dynamic_sweep_start_position) # starts sweeping blocks from here
    arm.safe_move_to_position(dynamic_sweep_end_position)   # end sweep from here
    arm.exec_gripper_cmd(0.04, 30)   
    gripper=arm.get_gripper_state()
    grip_dist=gripper["position"][0]+gripper["position"][1]
   
    if grip_dist < 0.025:                                   #check if block is grabbed
        arm.safe_move_to_position(dynamic_start_position)   #if not go back to start
    else:
        arm.safe_move_to_position(dynamic_start_position)   #if grabbed, move here 
        arm.safe_move_to_position(q)                        #move to stacking pos
        arm.exec_gripper_cmd(0.1, 30)                       #release block
        counter=1  ## if block is grabbed increment counter by 1


    return counter


if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()
    fk=FK()
    ik=IK()

    
    #([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    # all 4 cubes ([0.179206, -0.2,  0.01978261, -1.95, 0.1, 0.210353+pi/2, 0.75344866])
    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!




        ##### environmental parameters #######  
    if team =='blue':
        start_position = np.array([0.18, -0.2,  0.02, -1.95, 0.1, 0.210+pi/2, 0.72])  #empirically obtaind home position 
        

        target_b_d1=np.array([[ 0. , -1.  ,   0.,    0.5],         #HT matrix for 1st dynamic stack pos
        [-1. ,   0.  ,   0. ,   -0.21],
        [ 0. ,    0.  ,  -1. ,  0.23],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target_b_d1, success, rollout =ik.inverse(target_b_d1,start_position)   # solves Ik and returns q for target 
        
        target_b_d2=np.array([[ 0. , -1.  ,   0.,    0.5],         #HT matrix for 2nd dynamic stack pos
        [-1. ,   0.  ,   0. ,   -0.21],
        [ 0. ,    0.  ,  -1. ,  0.28],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target_b_d2, success, rollout =ik.inverse(target_b_d2,start_position)   # solves Ik and returns q for target

        target_b_d3=np.array([[ 0. , -1.  ,   0.,    0.5],          #HT matrix for 3rd dynamic stack pos
        [-1. ,   0.  ,   0. ,   -0.21],
        [ 0. ,    0.  ,  -1. ,  0.33],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target_b_d3, success, rollout =ik.inverse(target_b_d3,start_position)   # solves Ik and returns q for target

        target_b_d4=np.array([[ 0. , -1.  ,   0.,    0.5],          #HT matrix for 4th dynamic stack pos
        [-1. ,   0.  ,   0. ,   -0.21],
        [ 0. ,    0.  ,  -1. ,  0.38],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target_b_d4, success, rollout =ik.inverse(target_b_d4,start_position)    # solves Ik and returns q for target

        target_b_d5=np.array([[ 0. , -1.  ,   0.,    0.5],          #HT matrix for 5th dynamic stack pos
        [-1. ,   0.  ,   0. ,   -0.21],
        [ 0. ,    0.  ,  -1. ,  0.43],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target_b_d5, success, rollout =ik.inverse(target_b_d5,start_position)    # solves Ik and returns q for target

        q_DBlue=np.array([q_target_b_d1, q_target_b_d2, q_target_b_d3,q_target_b_d4,q_target_b_d5])  # array of stack positions for DBlocks

 

        target=np.array([[ 0. , -1.  ,   0.,    0.61871184],  #HT matrix for 1st static stack pos
        [-1. ,   0.  ,   0. ,   -0.15],
        [ 0. ,    0.  ,  -1. ,  0.23 ],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target, success, rollout =ik.inverse(target,start_position)   # solves Ik and returns q for target

        target2=np.array([[ 0. , -1.  ,   0.,    0.61871184],    #HT matrix for 2nd static stack pos
        [-1. ,   0.  ,   0. ,   -0.15],
        [ 0. ,    0.  ,  -1. ,  0.280 ],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target2, success, rollout =ik.inverse(target2,start_position)   # solves Ik and returns q for target
        
        
        target3=np.array([[ 0. , -1.  ,   0.,    0.61871184],  #HT matrix for 3rd static stack pos
        [-1. ,   0.  ,   0. ,   -0.15],
        [ 0. ,    0.  ,  -1. ,  0.33],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target3, success, rollout =ik.inverse(target3,start_position)   # solves Ik and returns q for target

        
        target4=np.array([[ 0. , -1.  ,   0.,    0.61871184],  #HT matrix for 4th static stack pos
        [-1. ,   0.  ,   0. ,   -0.15],
        [ 0. ,    0.  ,  -1. ,  0.38],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target4, success, rollout =ik.inverse(target4,start_position)    # solves Ik and returns q for target
    
        
        q_target=np.array([0.10825554,  0.38186018,-0.38261253, -1.82821353,  0.17022725, 2.17798181,  2.0092706 ])
        
        q_stack=np.array([q_target,q_target2, q_target3, q_target4])   # array of stack positions for SBlocks
        
        midjourney = np.array([             #mid way point to avoid collisions
            [0, -1, 0, 0.56], 
            [-1, 0, 0, 0],
            [0, 0, -1, 0.38],
            [0, 0, 0, 1],
        ])
        q_mid, success, rollout =ik.inverse(midjourney,start_position)   # solves Ik and returns q for mid way point
        
        
    ##### end environmental parameters #######  


    # STUDENT CODE HERE
        
        ## start picking dynamic blocks ####

        attempt=1
        dynamic_i=0
        while(dynamic_i<2):                                          # until 2 Dynamic blocks are placed 
            dynamic_i= dynamic_i + dynamic_blue(q_DBlue[dynamic_i])  # the dynamic_blue() returns 1 if if stacks a block successfully
            attempt+=1                                                  

        ## this while loop can also be chaged to consider number of attempts (sucessfull or not )instead of number of blockes placed 
        
        ## end picking dynamic blocks ####

        ## start static blocks ###

        arm.safe_move_to_position(start_position) # on your mark!
        arm.exec_gripper_cmd(0.12, 50)
        if team=='blue':
            d=1
        if team=='red':
            d=2
        q=arm.get_positions() 
        placed=0

        while(placed<4):    

            arm.safe_move_to_position(start_position)
            arm.exec_gripper_cmd(0.10, 20)

            ##read 
            q=arm.get_positions() 
            posss_array=[]
            for (name, pose) in detector.get_detections():  # detect block pos wrt cam frame
                print(name,'\n',pose)
                position = block_pos(pose, q) 
                print (position)
                if pose[1,3]>0:                             # if y coordinante of cube is negative ignote it (all blocks that are placed have negative y)
                    posss_array.append(position)

            ### changing cube HT matrix so that z axis point vetrtically down 
            for i in posss_array:       
                i[0:3,2] = np.array([0,0,-1])
                i[3,0:3] = np.array([0,0,0])
                i[3,3]=1
                i[1,3]=i[1,3]  
                i[0,0]=  0    
                i[1,0]= -1    
                i[0,1]= -1   
                i[1,1]= 0    
                i[2,0:2]=0

        #if block present then grab 
            if len(posss_array) > 0:
                q, success, rollout = ik.inverse(posss_array[0],start_position)
                if success:  
                    arm.safe_move_to_position(q)   ###move to block
                arm.exec_gripper_cmd(0.04, 65)    ## grab
                gripper=arm.get_gripper_state()
                grip_dist=gripper["position"][0]+gripper["position"][1]
                
                if grip_dist < 0.025:             # check if block is grabbed
                    arm.safe_move_to_position(start_position)   # if not go back to start pos
                    placed=0
                else:
                    arm.safe_move_to_position(q_mid)            # if grabbed go to mid way point to avoid collisions
                    arm.safe_move_to_position(q_stack[placed])  # then place at the stacking pos
                    arm.exec_gripper_cmd(0.1, 65)    ## release 

                    placed+=1

        ### end static blocks ###

        ##start picking 3 more dynaic blocks ###
        
        attempt=0
        while(attempt<3):
            dynamic_i= dynamic_i + dynamic_blue(q_DBlue[dynamic_i])
            attempt+=1


######## red team #########
### red team has eaxact same code as blue expect that some pos are mirrored ###

## follow blue team code for comments ####



    if team =='red':
        start_position = np.array([-0.18, -0.2,  0.02, -1.95, 0.1, 0.210+pi/2, 0.72])
        

        target_b_d1=np.array([[ 0. , -1.  ,   0.,    0.5],
        [-1. ,   0.  ,   0. ,   0.21],
        [ 0. ,    0.  ,  -1. ,  0.23],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target_b_d1, success, rollout =ik.inverse(target_b_d1,start_position)
        
        target_b_d2=np.array([[ 0. , -1.  ,   0.,    0.5],
        [-1. ,   0.  ,   0. ,   0.21],
        [ 0. ,    0.  ,  -1. ,  0.28],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target_b_d2, success, rollout =ik.inverse(target_b_d2,start_position)

        target_b_d3=np.array([[ 0. , -1.  ,   0.,    0.5],
        [-1. ,   0.  ,   0. ,   0.21],
        [ 0. ,    0.  ,  -1. ,  0.33],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target_b_d3, success, rollout =ik.inverse(target_b_d3,start_position)

        target_b_d4=np.array([[ 0. , -1.  ,   0.,    0.5],
        [-1. ,   0.  ,   0. ,   0.21],
        [ 0. ,    0.  ,  -1. ,  0.38],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target_b_d4, success, rollout =ik.inverse(target_b_d4,start_position)

        target_b_d5=np.array([[ 0. , -1.  ,   0.,    0.5],
        [-1. ,   0.  ,   0. ,   0.21],
        [ 0. ,    0.  ,  -1. ,  0.43],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target_b_d5, success, rollout =ik.inverse(target_b_d5,start_position)

        q_DBlue=np.array([q_target_b_d1, q_target_b_d2, q_target_b_d3,q_target_b_d4,q_target_b_d5])
            
    
### red team has eaxact same code as blue expect that some pos are mirrored ###

## follow blue team code for comments ####


        target=np.array([[ 0. , -1.  ,   0.,    0.61871184],
        [-1. ,   0.  ,   0. ,   0.15],
        [ 0. ,    0.  ,  -1. ,  0.23 ],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target, success, rollout =ik.inverse(target,start_position)

        target2=np.array([[ 0. , -1.  ,   0.,    0.61871184],
        [-1. ,   0.  ,   0. ,   0.15],
        [ 0. ,    0.  ,  -1. ,  0.280 ],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target2, success, rollout =ik.inverse(target2,start_position)
        
    
        target3=np.array([[ 0. , -1.  ,   0.,    0.61871184],
        [-1. ,   0.  ,   0. ,   0.15],
        [ 0. ,    0.  ,  -1. ,  0.33],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target3, success, rollout =ik.inverse(target3,start_position)

   
        target4=np.array([[ 0. , -1.  ,   0.,    0.61871184],
        [-1. ,   0.  ,   0. ,   0.15],
        [ 0. ,    0.  ,  -1. ,  0.38],
        [ 0. ,    0.  ,   0.  ,  1.        ]])
        q_target4, success, rollout =ik.inverse(target4,start_position)
    
        
        #q_target=np.array([0.10825554,  0.38186018,-0.38261253, -1.82821353,  0.17022725, 2.17798181,  2.0092706 ])
        
        q_stack=np.array([q_target,q_target2, q_target3, q_target4])
        midjourney = np.array([
            [0, -1, 0, 0.56],
            [-1, 0, 0, 0],
            [0, 0, -1, 0.38],
            [0, 0, 0, 1],
        ])
        q_mid, success, rollout =ik.inverse(midjourney,start_position)
      
        
    ##### end environmental parameters #######  



    # STUDENT CODE HERE
        attempt=1
        dynamic_i=0
        while(attempt<3):
            dynamic_i= dynamic_i + dynamic_red(q_DBlue[dynamic_i])
            attempt+=1
        
        arm.safe_move_to_position(start_position) # on your mark!
        arm.exec_gripper_cmd(0.12, 50)
        if team=='blue':
            d=1
        if team=='red':
            d=2
        q=arm.get_positions() 
    
        counter=0
        placed=0      
        

        while(placed<4):    

            arm.safe_move_to_position(start_position)
            arm.exec_gripper_cmd(0.10, 20)

            ##read 
            q=arm.get_positions() 
            posss_array=[]
            for (name, pose) in detector.get_detections():  # detect block pos wrt cam frame
                print(name,'\n',pose)
                position = block_pos(pose, q) 
                print (position)
                if pose[1,3]>0:                             # if y coordinante of cube is negative ignote it (all blocks that are placed have negative y)
                    posss_array.append(position)

            ### changing cube HT matrix so that z axis point vetrtically down 
            for i in posss_array:       
                i[0:3,2] = np.array([0,0,-1])
                i[3,0:3] = np.array([0,0,0])
                i[3,3]=1
                i[1,3]=i[1,3]  
                i[0,0]=  0    
                i[1,0]= -1    
                i[0,1]= -1   
                i[1,1]= 0    
                i[2,0:2]=0

        #if block present then grab 
            if len(posss_array) > 0:
                q, success, rollout = ik.inverse(posss_array[0],start_position)
                if success:  
                    arm.safe_move_to_position(q)   ###move to block
                arm.exec_gripper_cmd(0.04, 65)    ## grab
                gripper=arm.get_gripper_state()
                grip_dist=gripper["position"][0]+gripper["position"][1]
                
                if grip_dist < 0.025:             # check if block is grabbed
                    arm.safe_move_to_position(start_position)   # if not go back to start pos
                    placed=0
                else:
                    arm.safe_move_to_position(q_mid)            # if grabbed go to mid way point to avoid collisions
                    arm.safe_move_to_position(q_stack[placed])  # then place at the stacking pos
                    arm.exec_gripper_cmd(0.1, 65)    ## release 

                    placed+=1
        

        attempt=0
        while(attempt<3):
            dynamic_i= dynamic_i + dynamic_red(q_DBlue[dynamic_i])
            attempt+=1



    
    
    # # END STUDENT CODE
