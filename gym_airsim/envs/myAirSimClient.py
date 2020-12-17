import numpy as np
import time
import math
import cv2
from pylab import array, arange, uint8 
from PIL import Image
import eventlet
from eventlet import Timeout
import multiprocessing as mp
import sys
# Change the path below to point to the directoy where you installed the AirSim PythonClient
sys.path.append('/home/mojtaba/AirGym-master')

from AirSimClient import *


class myAirSimClient(MultirotorClient):

    def __init__(self):  

      
        self.img1 = None
        self.img2 = None

        """self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)"""""

        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        self.enableApiControl(True)
        self.takeoffAsync().join()
        self.armDisarm(True)
    
        #self.home_pos = self.getMultirotorState().kinematics_estimated.position #getPosition()
        #print("home_pos : ",self.home_pos)
        #self.home_ori = self.getMultirotorState().kinematics_estimated.orientation
        #print("home_ori : ",self.home_ori)
        
        self.z = -6
    
    def straight(self, duration, speed):
        #pitch, roll, yaw  = self.toEulerianAngle(self.simGetVehiclePose().orientation) #self.getPitchRollYaw()
        pitch, roll, yaw = airsim.to_eularian_angles(self.getMultirotorState().kinematics_estimated.orientation)
        #print("yaw in straight : ", yaw)
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        self.moveByVelocityZAsync(vx, vy, self.z, duration, DrivetrainType.ForwardOnly).join()
        start = time.time()
        return start, duration
    
    def yaw_right(self, duration):
        self.rotateByYawRateAsync(30, duration).join()
        start = time.time()
        return start, duration
    
    def yaw_left(self, duration):
        self.rotateByYawRateAsync(-30, duration).join()
        start = time.time()
        return start, duration
    
    
    def take_action(self, action):
		
        #check if copter is on level cause sometimes he goes up without a reason
        x = 0
        #while self.getMultirotorState().z_val < -7.0: #airgym.getMultirotorState().kinematics_estimated_position
        while self.getMultirotorState().kinematics_estimated.position.z_val < -7.0:
            print("alt adjust")
            #self.moveToZAsync(-6, 3)
            self.moveToZAsync(-6, 3).join()
            time.sleep(1)
            #print(self.getMultirotorState().kinematics_estimated.position.z_val, "and", x)
            x = x + 1
            if x > 10:
                return True        
        
    
        start = time.time()
        duration = 0 
        
        collided = False

        if action == 0:

            start, duration = self.straight(1, 4)

        
            #while duration > time.time() - start:
            if self.simGetCollisionInfo().has_collided == True:
                return True
                
            self.moveByVelocityAsync(0, 0, 0, 0.1).join()
            self.rotateByYawRateAsync(0, 0.1).join()
            
            
        if action == 1:
         
            start, duration = self.yaw_right(1)
            
            #while duration > time.time() - start:
            if self.simGetCollisionInfo().has_collided == True:
                return True
            
            self.moveByVelocityAsync(0, 0, 0, 0.1).join()
            self.rotateByYawRateAsync(0, 0.1).join()
            
        if action == 2:
            
            start, duration = self.yaw_left(1)
            
            #while duration > time.time() - start:
            if self.simGetCollisionInfo().has_collided == True:
                return True
                
            self.moveByVelocityAsync(0, 0, 0, 1).join()
            self.rotateByYawRateAsync(0, 0.1).join()
            
        return collided
    
    def goal_direction(self, goal, pos):
        #print(self.getMultirotorState().kinematics_estimated.orientation)
        pitch, roll, yaw  =  airsim.to_eularian_angles(self.getMultirotorState().kinematics_estimated.orientation) #self.getPitchRollYaw()

        #pitch, roll, yaw = airsim.to_eularian_angles(self.simGetVehiclePose().orientation)
        yaw = math.degrees(yaw) 
        
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
        
        return ((math.degrees(track) - 180) % 360) - 180

    def random_crop(self, image, out_height=30, out_width=100):
        h, w, c = image.shape  # assuming image shape is (h, w, c)
        h_max_crop = h - out_height + 1
        w_max_crop = w - out_width + 1
        h1 = np.random.randint(0, h_max_crop)
        w1 = np.random.randint(0, w_max_crop)
        cropped_image = image[h1:h1 + out_height, w1:w1+out_width,:]
        return cropped_image
    
    def getScreenDepthVis(self, track):
        try:
            responses = self.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)])
        except Exception as e:
            print('EXCEEEEPTION')
            #self.AirSim_reset()
            # time.sleep(2)
            responses = self.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)])  # airsim.ImageType.Scene , False, False



        '''response = responses[0]
        my_img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        print('h: ',responses[0].height )
        print(' w: ',responses[0].width )

        #print(responses[0].height * responses[0].width * 3)

        if my_img1d.size == responses[0].height * responses[0].width*3:
            # reshape image
            img_rgb = my_img1d.reshape(response.height, response.width, 3)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        else:
            print("image broken ")
            gray = np.ones((144, 256))'''




        #responses = self.simGetImages([ImageRequest(0, AirSimImageType.DepthPerspective, True, False)])
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        #img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        if img1d.size == responses[0].height * responses[0].width:
            # reshape image
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        else:
            print("image broken ")
            img2d = np.ones((144, 256))


        image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))

        factor = 10
        maxIntensity = 255.0 # depends on dtype of image data
        
        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
        newImage1 = (maxIntensity)*(image/maxIntensity)**factor
        newImage1 = array(newImage1,dtype=uint8)
        #newImage1 =  gray
        small = cv2.resize(newImage1, (0,0), fx=0.39, fy=0.38)
        small_ = cv2.resize(newImage1, (100, 30), interpolation = cv2.INTER_AREA)
        cut = small[20:40,:]
        '''info_section = np.zeros((10, cut.shape[1]), dtype=np.uint8) + 255
        info_section[9, :] = 0

        line = np.int((((track - -180) * (100 - 0)) / (180 - -180)) + 0)

        if line != (0 or 100):
            info_section[:, line - 1:line + 2] = 0
        elif line == 0:
            info_section[:, 0:3] = 0
        elif line == 100:
            info_section[:, info_section.shape[1] - 3:info_section.shape[1]] = 0'''
       ###############################################################################
        info_section = np.zeros((10,cut.shape[1]),dtype=np.uint8) - 255
        info_section[9,:] = 255
        
        line = np.int((((track - -180) * (100 - 0)) / (180 - -180)) + 0)
        
        if line != (0 or 100):
            info_section[:,line-4:line+5]  = 255
        elif line == 0:
            info_section[:,0:3]  = 255
        elif line == 100:
            info_section[:,info_section.shape[1]-3:info_section.shape[1]]  = 255
        ########################################################################################
        total = np.concatenate((info_section, cut), axis=0)


            
        cv2.imshow("Test", total)
        cv2.waitKey(30)
        
        return total


    def AirSim_reset(self):
        #print("resssssssssssssseeeet")
        self.reset()
        #time.sleep(2)
        self.enableApiControl(True)
        self.armDisarm(True)
        #time.sleep(2)
        self.takeoffAsync().join()
        self.moveToZAsync(self.z, 3).join()
        #time.sleep(3)
        
    
    def AirSim_reset_old(self):
        
        reset = False
        #z = -6.0
        while reset != True:

            now = self.getMultirotorState().kinematics_estimated.position
            self.simSetVehiclePose(Pose(Vector3r(now.x_val, now.y_val, -30),Quaternionr(self.home_ori.w_val, self.home_ori.x_val, self.home_ori.y_val, self.home_ori.z_val)), True)
            now = self.getMultirotorState().kinematics_estimated.position
            
            if (now.z_val - (-30)) == 0:
                self.simSetVehiclePose(Pose(Vector3r(self.home_pos.x_val, self.home_pos.y_val, -30),Quaternionr(self.home_ori.w_val, self.home_ori.x_val, self.home_ori.y_val, self.home_ori.z_val)), True)
                now =self.getMultirotorState().kinematics_estimated.position
                
                if (now.x_val - self.home_pos.x_val) == 0 and (now.y_val - self.home_pos.y_val) == 0 and (now.z_val - (-30)) == 0 :
                    self.simSetVehiclePose(Pose(Vector3r(self.home_pos.x_val, self.home_pos.y_val, self.home_pos.z_val),Quaternionr(self.home_ori.w_val, self.home_ori.x_val, self.home_ori.y_val, self.home_ori.z_val)), True)
                    now = self.getMultirotorState().kinematics_estimated.position
                    
                    if (now.x_val - self.home_pos.x_val) == 0 and (now.y_val - self.home_pos.y_val) == 0 and (now.z_val - self.home_pos.z_val) == 0:
                        reset = True
                        self.moveByVelocityAsync(0, 0, 0, 1)
                        time.sleep(1)
                        
        self.moveToZAsync(self.z, 3)
        time.sleep(3)
