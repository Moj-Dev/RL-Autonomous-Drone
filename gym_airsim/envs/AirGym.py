import logging
import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
from gym.spaces.box import Box
from numpy.linalg import norm
from gym_airsim.envs.myAirSimClient import *
        
from AirSimClient import *

logger = logging.getLogger(__name__)


class AirSimEnv(gym.Env):

    airgym = None
        
    def __init__(self):
        # left depth, center depth, right depth, yaw
        self.observation_space = spaces.Box(low=0, high=255, shape=(30, 100))
        self.state = np.zeros((30, 100), dtype=np.uint8)  
        
        self.action_space = spaces.Discrete(3)
		
        self.goal = 	[180.0, 0.0] # global xy coordinates
        
        
        self.episodeN = 0
        self.stepN = 0
        self.stepN_Sum=0
        self.stepN_Avg=0
        
        self.allLogs = { 'reward':[0] }
        self.allLogs['distance'] = [221]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]
        self.success_rate_cnt =0
        self.success_rate = 0
        self.success_rate_done_cnt=0


        self._seed()
        
        global airgym
        airgym = myAirSimClient()

        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def my_computeReward(self, now, track_now):
        p1 = np.array(list((0, 0)))  # Starting Position
        p2 = np.array(list((self.goal[0], self.goal[1])))  # Goal Position
        p3 = np.array(list((now.x_val, now.y_val)))  # Currnet Position
        #ct_d = np.cross(p2 - p1, p1 - p3)  # Cross-track distance
        ct_d = norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)  # Cross-track distance
        ct_d = 0.075*ct_d
        distance_now = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        distance_before = self.allLogs['distance'][-1]
        r = -1
        r = r + (distance_before - distance_now) - ct_d
        #print("Cross-track distance",ct_d, "     reward:",r)

        return r, distance_now







    def computeReward(self, now, track_now):
	
		# test if getPosition works here liek that
		# get exact coordiantes of the tip
      
        distance_now = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        distance_before = self.allLogs['distance'][-1]
        r = -1
        
        """
        if abs(distance_now - distance_before) < 0.0001:
            r = r - 2.0
            #Check if last 4 positions are the same. Is the copter actually moving?
            if self.stepN > 5 and len(set(self.allLogs['distance'][len(self.allLogs['distance']):len(self.allLogs['distance'])-5:-1])) == 1: 
                r = r - 50
        """  
            
        r = r + (distance_before - distance_now)
            
        return r, distance_now
		
    
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.addToLog('action', action)
        
        self.stepN += 1


        collided = airgym.take_action(action)

        now = airgym.getMultirotorState().kinematics_estimated.position #getPosition()
        #track = airgym.goal_direction(self.goal, now) 
######################################################################################################################################
        pos = airgym.getMultirotorState().kinematics_estimated.position
        #print("kinematik in step : ",airgym.getMultirotorState().kinematics_estimated)
        goal = self.goal

        pitch, roll, yaw  =  airsim.to_eularian_angles(airgym.getMultirotorState().kinematics_estimated.orientation) #self.getPitchRollYaw()
        yaw = math.degrees(yaw)
        #print("   pos x : ", pos.x_val," pos y : ", pos.y_val, " yaw: ", yaw," goal : ", goal)
        print("[",pos.x_val,",",pos.y_val,"]",",")

        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
        
        track = ((math.degrees(track) - 180) % 360) - 180    
######################################################################################################################################


        if collided == True:
            done = True
            #print("<<<<<<<<<<<<<<   COLLISION     >>>>>>>>>>>>>>>>>")
            reward = -100.0
            distance = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        elif collided == 99:
            done = True
            #print("<<<<<<<<<<<<<<   99     >>>>>>>>>>>>>>>>>")

            reward = 0.0
            distance = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        else: 
            done = False
            reward, distance = self.my_computeReward(now, track)
            #reward, distance = self.computeReward(now, track)

        
        # Youuuuu made it
        if distance < 3:
            #print("<<<<<<<<<<<<<<   SUCESS     >>>>>>>>>>>>>>>>>")
            done = True
            self.success_rate_done_cnt+=1
            reward = 100.0
        
        self.addToLog('reward', reward)
        rewardSum = np.sum(self.allLogs['reward'])
        self.addToLog('distance', distance)
        self.addToLog('track', track)      
            
        # Terminate the episode on large cumulative amount penalties, 
        # since drone probably got into an unexpected loop of some sort
        if rewardSum <= -100:
            done = True
        
        #sys.stdout.write("\r\x1b[K{}/{}==>reward/depth: {:.1f}/{:.1f}   \t {:.0f}  {:.0f}".format(self.episodeN, self.stepN, reward, rewardSum, track, action))
        #sys.stdout.flush()



        info = {"x_pos" : now.x_val, "y_pos" : now.y_val}
        self.state = airgym.getScreenDepthVis(track)

        return self.state, reward, done, info

    def addToLog (self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)
        
    def _reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """

        airgym.AirSim_reset()
        #print("after reset func")
        self.stepN_Sum+=self.stepN
        if self.success_rate_cnt >= 50:
            self.stepN_Avg=self.stepN_Sum/50
            self.success_rate = self.success_rate_done_cnt / 50
            self.success_rate_done_cnt = 0
            self.success_rate_cnt = 0
            self.stepN_Sum=0
            print("<<<<<<<<<<    Sucess Rate: ", self.success_rate, "  Step AVG: ", self.stepN_Avg  ,"    >>>>>>>>>>")

        self.stepN = 0
        self.episodeN += 1
        self.success_rate_cnt += 1

        
        self.allLogs = { 'reward': [0] }
        self.allLogs['distance'] = [221]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]
        
        #print("")
        
        now = airgym.getMultirotorState().kinematics_estimated.position #getPosition()
        #print(airgym.getMultirotorState())
        #print("orinetation : ",airgym.getMultirotorState().kinematics_estimated.orientation)
        #print("position : ",airgym.getMultirotorState().kinematics_estimated.position)

        track = airgym.goal_direction(self.goal, now)
        self.state = airgym.getScreenDepthVis(track)
        
        return self.state
