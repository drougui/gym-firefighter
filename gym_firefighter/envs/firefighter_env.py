"""
A firefighter robot
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time


logger = logging.getLogger(__name__)

class FirefighterEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        #self.gravity = 9.8
        #self.masscart = 1.0
        #self.masspole = 0.1
        #self.total_mass = (self.masspole + self.masscart)
        #self.length = 0.5 # actually half the pole's length
        #self.polemass_length = (self.masspole * self.length)
        #self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
	#Initial datas
	self.vit = 1
	self.rota_speed = 1
        self.treeslocations = [1.0,3.0, 5.0, 8.0, 2.0, 4.0, 6.0, 7.0];
        self.battery_zone = [2.0, 7.0];
        self.water_zone = [7.0,1.0];
        self.fuites_view = [625, 675, 625, 675, 625, 675, 625, 675, 425,425,475,475,525,525,575,575];
        self.fuites = [0,0,0,0,0,0,0,0];
	#Initilisation of ending counters
	self.fuites_count = 0
	self.count = 0
	self.counter = 0
	self.reservoir = 100
	self.water_reserve = 1000
	self.battery = 10000000000000000
	self.temperature = 0
	self.stepcount = 50
	self.rand = 0
	#limits
	self.tempLim = 200
	self.battery_lim = 10000000000000000
	self.water_reserve_lim = 1000
	self.reservoir_lim = 100	
	self.fuite_lim= len(self.fuites)/2
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds$
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max]+[0]*(len(self.treeslocations)/2))


        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.reset()
        self.viewer = None

        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
	screen_width = 600
        screen_height = 600
	
        world_width = self.x_threshold*2
        scale = screen_width/world_width
        cartwidth = 20.0
        cartheight = 20.0
	x_lim = 0.01*cartwidth*scale/screen_width
	y_lim = 0.01*cartheight*scale/screen_height


        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state

	#water loss via nb of fuites
	Frand = self.np_random.uniform(0,1)	
	if (Frand<0.05 and self.fuites_count < len(self.fuites)): #apparition de fuite aleatoire
		self.fuites_count += 1
		self.fuites[self.fuites_count - 1] = 1
	
	if (self.water_reserve >0): 
		loss = np.floor(self.np_random.uniform(0,1)*5)*self.fuites_count #perte d'eau
		self.water_reserve -= loss
		if (self.water_reserve < 0):
			self.water_reserve = 0
		
	#maintenance of the water reserve
	if (self.stepcount == 50):
		self.rand = self.np_random.uniform(0,1)
		
	if (self.rand > 0.5): #detection aleatoire (ici une chance sur deux)
		Hrand = self.np_random.uniform(0,1)
		if (Hrand > 0.5 or self.stepcount < 50): # une chance sur deux d'enclencher une action, puis effectuees au bout d'un temps aleatoire
			self.stepcount -= np.floor(self.np_random.uniform(0,1)*10)
		if (self.stepcount <= 0):
	 		self.stepcount = 50
	 		self.water_reserve += 50 #gain d'eau dans la reserve lors de l'action effectuee
	 		if (self.water_reserve > self.water_reserve_lim):
	 			self.water_reserve = self.water_reserve_lim
	 		if (self.fuites_count >0):
	 			self.fuites_count -= 1 #diminution du niveau de fuite de 1
	 			self.fuites[self.fuites_count] = 0
			
	

	#tree states
        x = state[0]
        y = state[1]
        theta = state[2]
        theta_dot = state[3]
        treesStatesList = []
        for i in range(len(self.treeslocations)/2):
            treesStatesList.append(state[4+i])
	treesStates = ()

	#movement
	vitesse= 0
	if (action == 1):
		vitesse = self.vit
	elif (action == 0):
		vitesse = -self.vit

	vitesse_rota = 0
	if (action == 3):
		vitesse_rota = self.rota_speed
	elif (action == 4):
		vitesse_rota = -self.rota_speed
	#vitesse = 1
	theta = theta + self.tau*vitesse_rota*10
	
	sintheta = np.sin(theta)
	costheta = np.cos(theta)
        x  = x + self.tau * vitesse* costheta
	y = y + self.tau * vitesse * sintheta


	#filling zone
	#battery
	if ( (0.1*self.battery_zone[0]-x_lim<self.state[0]*scale/screen_width+0.5<0.1*self.battery_zone[0]+x_lim) and (0.1*self.battery_zone[1]-y_lim<self.state[1]*scale/screen_width+0.5<0.1*self.battery_zone[1]+y_lim) and self.battery<self.battery_lim):
		self.battery += 1
		if (self.battery > self.battery_lim):
			self.battery = self.battery_lim
	
	#water
	if ( (0.1*self.water_zone[0]-x_lim<self.state[0]*scale/screen_width+0.5<0.1*self.water_zone[0]+x_lim) and (0.1*self.water_zone[1]-y_lim<self.state[1]*scale/screen_width+0.5<0.1*self.water_zone[1]+y_lim) and self.reservoir<self.reservoir_lim and self.water_reserve > 0):
		if (self.reservoir >= self.reservoir_lim):
			self.reservoir = self.reservoir_lim
		else :
			self.reservoir += 1
			self.water_reserve -=1
		
	
	#temperature indicator
	#zone de detection => cercle
	tempcounter = 0
	for i in range(len(self.treeslocations)/2):
		ifcounter = 0
		d = np.sqrt(np.power((np.absolute(0.1*self.treeslocations[i]))-(np.absolute(self.state[0]*scale/screen_width+0.5)),2)+np.power((np.absolute(0.1*self.treeslocations[i+(len(self.treeslocations)/2)])-np.absolute(self.state[1]*scale/screen_width+0.5)),2))
		if (self.temperature != 0 and d > 0.05 and ifcounter == 0):
			ifcounter +=1
			tempcounter += 1
		elif (d<0.05 and self.state[4+i]==0 and self.temperature != 0 and ifcounter == 0):
			tempcounter += 1

	#increase or decrease of the robot temperature
	if (tempcounter == len(self.treeslocations)/2 and self.temperature != 0):
		self.temperature -= 1
	for i in range(len(self.treeslocations)/2):
		d = np.sqrt(np.power((np.absolute(0.1*self.treeslocations[i]))-(np.absolute(self.state[0]*scale/screen_width+0.5)),2)+np.power((np.absolute(0.1*self.treeslocations[i+(len(self.treeslocations)/2)])-np.absolute(self.state[1]*scale/screen_width+0.5)),2))
		if (d<0.05 and self.state[4+i]==1):
			self.temperature += 1
	#decrease of the initial values due to actions
	if (action == 0 or action == 1 or action == 2 or action == 3 or action == 4):
		self.battery -= 1
	if (action ==2):
		self.reservoir -= 1
	
	#action of extinguishing the fire
	for i in range(len(self.treeslocations)/2):
		d = np.sqrt(np.power((np.absolute(0.1*self.treeslocations[i]))-(np.absolute(self.state[0]*scale/screen_width+0.5)),2)+np.power((np.absolute(0.1*self.treeslocations[i+(len(self.treeslocations)/2)])-np.absolute(self.state[1]*scale/screen_width+0.5)),2))
		if (d<0.05 and self.state[4+i]==1 and action == 2):
			treesStatesList[i] = 0
			self.counter += 1			

	
	#state of trees
        for i in range(len(self.treeslocations)/2):
            b = treesStatesList[i]
            if((treesStatesList[i]==0) and (self.np_random.uniform(0,1)>0.95)):
                b=1
            treesStates += (b,)
        self.state = (x,y,theta,theta_dot) + treesStates 
	

        #time.sleep(0.5)
	self.count += 1
	done = False
	
	#countermeasures actions
	
	if (action == 5):
		if (self.reservoir <= self.reservoir_lim*0.2):
			print("Low robot water")
		if (self.battery <= self.battery_lim*0.2):
			print("Pay Attention!! Low Battery")
		if (self.water_reserve<= self.reservoir_lim):
			print("Pay Attention!! Low Water Reserve")

	#ending conditions
	if (self.temperature == self.tempLim):
		done = True
		reward = self.counter
		return np.array(self.state), reward, done, {}
	elif (self.battery == 0):
		done = True
		reward = self.counter
		return np.array(self.state), reward, done, {}
	elif (self.reservoir == 0):
		done = True
		reward = self.counter
		return np.array(self.state), reward, done, {}
	elif (self.count == 20000):
		done = True
		reward = self.counter
        
	reward = self.counter
	return np.array(self.state), reward, done, {}

    def _reset(self):
        treesStates = []
        for i in range(len(self.treeslocations)/2):
            treesStates.append(0)
        self.state = np.concatenate( (self.np_random.uniform(low=-0.05, high=0.05, size=(4,)),treesStates), axis=0)
	self.counter = 0
	self.count = 0
	self.temperature = 0
	self.battery = self.battery_lim
	self.reservoir = self.reservoir_lim
	self.water_reserve = self.water_reserve_lim
	self.stepcount = 50
	self.fuites = [0,0,0,0,0,0,0,0];
	self.fuites_count = 0
	print(self.state)
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 700
        screen_height = 600

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 20.0
        cartheight = 20.0
	
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            
            #affichage robot
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
	    self.chose = []
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
	    self.chose.append(cart)
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(self.chose[0])
            
             #affichage ligne de separation
            self.ligne = rendering.Line((600,0), (600,600))
            self.ligne.set_color(0,0,0)
            self.viewer.add_geom(self.ligne)
            
            #affichage arbres
            self.truc = []
            cpt = 0
            for i in range(len(self.treeslocations)/2):
		tree_x = self.treeslocations[i]
		tree_y = self.treeslocations[i+len(self.treeslocations)/2]
                self.truc.append(rendering.FilledPolygon([(tree_x*screen_width/10-10,tree_y*screen_height/10-10), (tree_x*screen_width/10+10,tree_y*screen_height/10-10), (tree_x*screen_width/10+10,tree_y*screen_height/10+10), (tree_x*screen_width/10-10,tree_y*screen_height/10+10)]))
                self.truc[cpt].set_color(0,1,0)
                self.viewer.add_geom(self.truc[cpt])
                cpt+=1
             
            #affichage fuites
            self.fuites_render = []   
            cpt = 0
            for i in range(len(self.fuites)):
		fuite_x = self.fuites_view[i]
		fuite_y = self.fuites_view[i+len(self.fuites_view)/2]
                self.fuites_render.append(rendering.FilledPolygon([(fuite_x-25,fuite_y-25), (fuite_x-25,fuite_y+25), (fuite_x+25,fuite_y-25), (fuite_x+25,fuite_y+25)]))
                self.fuites_render[cpt].set_color(0,0,0)
                self.viewer.add_geom(self.fuites_render[cpt])
                cpt+=1
                
            #affichage zone batterie
            l,r,t,b = -cartwidth,cartwidth,cartheight,-cartheight
            zone_b = rendering.FilledPolygon([(self.battery_zone[0]*screen_width/10+l,self.battery_zone[1]*screen_height/10+b), (self.battery_zone[0]*screen_width/10+l,self.battery_zone[1]*screen_height/10+t), (self.battery_zone[0]*screen_width/10+r,self.battery_zone[1]*screen_height/10+t), (self.battery_zone[0]*screen_width/10+r,self.battery_zone[1]*screen_height/10+b)])
            zone_b.set_color(1,1,0)
            self.viewer.add_geom(zone_b)
            
            #affichage zone eau
            self.zone_w = rendering.FilledPolygon([(self.water_zone[0]*screen_width/10+l,self.water_zone[1]*screen_height/10+b), (self.water_zone[0]*screen_width/10+l,self.water_zone[1]*screen_height/10+t), (self.water_zone[0]*screen_width/10+r,self.water_zone[1]*screen_height/10+t), (self.water_zone[0]*screen_width/10+r,self.water_zone[1]*screen_height/10+b)])
            self.zone_w.set_color(0,0,1)
            self.viewer.add_geom(self.zone_w)
            
            #affichage etat reservoir robot
            l,r,t,b = -10,10,10,-10
            self.reservoir_render = rendering.FilledPolygon([(650+l, 325+b),(650+l, 325+t),(650+r, 325+t),(650+r,325+b)])
            self.reservoir_render.set_color(0,0,1)
            self.viewer.add_geom(self.reservoir_render)
            
                
                
        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
	carty = x[1]*scale+screen_height/2
	self.carttrans.set_rotation(x[2])
        self.carttrans.set_translation(cartx, carty)
        #changement couleur arbre
        for i in range(len(self.treeslocations)/2):
            if(x[i+4]==1):
                self.truc[i].set_color(1,0,0)
            else:
		self.truc[i].set_color(0,1,0)
	
	#changement couleur fuite
	for i in range(len(self.fuites)):
            if(self.fuites[i]==1):
                self.fuites_render[i].set_color(0,0,1)
            else:
		self.fuites_render[i].set_color(0,0,0)
		
	#le probleme est juste ici.
	reservoir_color = (self.reservoir_lim-self.reservoir)/self.reservoir_lim
	print(self.reservoir_lim)
	print(self.reservoir)
	print(self.reservoir/self.reservoir_lim)
	print(reservoir_color)
	self.reservoir_render.set_color(reservoir_color,reservoir_color,1)
	
	#changement couleur cart selon la temperature
	red_color = self.temperature /self.tempLim
	self.chose[0].set_color(red_color,0,0)
	#changement couleur zone eau
	blue_color = 1-(self.water_reserve/self.water_reserve_lim)
	print(blue_color)
	self.zone_w.set_color(blue_color,blue_color,1)
	#print("hello world")
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

