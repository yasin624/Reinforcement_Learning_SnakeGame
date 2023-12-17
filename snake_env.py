import gym

from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import  deque
from tqdm import tqdm



SNAKE_LEN_GOAL=50  # bŕ oyundak'hamle sayisi
class snake_game(gym.Env):

    def __init__(self,progrest_total=100,render_model="human"):
        super(snake_game,self).__init__()
        self.action_space=spaces.Discrete(4)   # seçim yapabilme değeri
        self.progrest=tqdm(total=progrest_total)
        self.observation_space=spaces.Box(low=-500,high=500,shape=(6+SNAKE_LEN_GOAL,),dtype=np.float32)

        self.render_mode=render_model


    def step(self,Action):
        self.prev_actions.append(Action)  # hamle tutucu son 50 hamleyi tutar


        if self.render_mode=="human":
            cv2.imshow('a', self.img)
            self.img = np.zeros((500, 500, 3), dtype='uint8')
            # Display Apple
            cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]), (self.apple_position[0] + 10, self.apple_position[1] + 10),
                          (246, 160, 180), -1)

            # Display Snake
            for position in self.snake_position:
                cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 255, 0), -1)
                cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 0, 0), 2)

        # Takes step after fixed time
        t_end = time.time() + 0.2
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(125)
            else:
                continue



        # Change the head position based on the button direction
        if Action == 1:
            self.snake_head[0] += 10
        elif Action == 0:
            self.snake_head[0] -= 10
        elif Action == 2:
            self.snake_head[1] += 10
        elif Action == 3:
            self.snake_head[1] -= 10

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = self.collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
            self.reward+=5
            self.pased=True

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        if self.collision_with_boundaries(self.snake_head) == 1 or self.collision_with_self(self.snake_position) == 1:


            if self.render_mode=="human":
                font = cv2.FONT_HERSHEY_SIMPLEX
                self.img = np.zeros((500, 500, 3), dtype='uint8')
                cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('a', self.img)

            self.done=True

        # ÖDÜL VEYA CEVA DEĞERLERİ
        self.Sx = self.snake_head[0]
        self.Sy = self.snake_head[1]
        self.Ax = self.apple_position[0]
        self.Ay = self.apple_position[1]

        self.S_leng = len(self.snake_position) - 3

        calculation=self.calculate_distance(self.snake_head, self.apple_position)

        if self.pased:
            self.pased=False
            self.S_To_A_calculation=calculation

        elif self.S_To_A_calculation > calculation :
            self.reward+=0.1

            self.S_To_A_calculation=calculation

        else:
            self.reward-=0.1
            self.S_To_A_calculation=calculation


        if self.done:
            self.reward-=100
            self.progrest.update(1)

        """print("="*100)
        print("penalty : ",self.penalty)
        print("prize : ",self.prize)"""

        self.observation = np.array([self.Sx, self.Sy, self.Ax, self.Ay, self.S_leng, self.S_To_A_calculation] + list(self.prev_actions),dtype=np.float32)




        return  self.observation,np.array(self.reward,dtype=np.float32),np.array(self.done,dtype=np.float32)





    def reset(self):
        self.done=False

        if self.render_mode=="human": self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.reward=0
        self.pased=False


        self.snake_head = [250, 250]

        # ÖDÜL VEYA CEVA DEĞERLERİ
        self.Sx=self.snake_head[0]
        self.Sy = self.snake_head[1]
        self.Ax = self.apple_position[0]
        self.Ay = self.apple_position[1]

        self.S_leng=len(self.snake_position)-3

        self.S_To_A_calculation=self.calculate_distance(self.snake_head,self.apple_position)

        self.prev_actions=deque(maxlen=SNAKE_LEN_GOAL)
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        self.observation= [self.Sx, self.Sy, self.Ax,self.Ay, self.S_leng,self.S_To_A_calculation]+list(self.prev_actions)#[-1 for i in range(SNAKE_LEN_GOAL)]


        return np.array(self.observation,dtype=np.float32)

    def calculate_distance(self,point1, point2):
        """
        İki nokta arasındaki Euclidean mesafeyi hesaplar.

        Parameters:
        point1 (numpy.ndarray): İlk noktanın koordinatları [x1, y1]
        point2 (numpy.ndarray): İkinci noktanın koordinatları [x2, y2]

        Returns:
        float: İki nokta arasındaki mesafe
        """
        distance = np.sqrt(((point2[0] - point1[0]) ** 2)+((point2[1] - point1[1]) ** 2))
        return distance
    def collision_with_apple(self,apple_position, score):
        apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
        score += 1
        return apple_position, score

    def collision_with_boundaries(self,snake_head):
        if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0 :
            return 1
        else:
            return 0

    def collision_with_self(self,snake_position):
        snake_head = snake_position[0]
        if snake_head in snake_position[1:]:
            return 1
        else:
            return 0





