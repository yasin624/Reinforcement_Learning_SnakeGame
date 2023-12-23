
import gym

from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import  deque
from tqdm import tqdm



ARENA_size=(500, 500, 3)
ARENA_MASK_size=np.array((500, 500, 1),dtype=np.int32)
OBJ_SİZE=10
SİZE=(ARENA_MASK_size[:2]/OBJ_SİZE).astype("int32")

class snake_game(gym.Env):

    def __init__(self,render_mode="human",Sk=1.55,Pk=0.1,Tk=0.1,Rk=1.1):
        super(snake_game,self).__init__()
        
        self.Sk=Sk
        self.Pk=Pk
        self.Tk=Tk
        self.Rk=Rk
        
        self.action_space=spaces.Discrete(4)   # seçim yapabilme değeri
        self.observation_space=spaces.Box(low=-ARENA_size[0],high=ARENA_size[1],shape=(7+(80*80),),dtype=np.float32)

        self.render_mode=render_mode
    def step(self,Action):
        step_reward=0
        if self.render_mode=="human":
            self.img = np.zeros(ARENA_size, dtype='uint8')
            font = cv2.FONT_HERSHEY_SIMPLEX

            info=np.zeros((50,self.img.shape[1],self.img.shape[2]))
            info[:,:]=[54,54,54]
            cv2.putText(info, 'ScoreQ : {}  |'.format(self.score), (50, 30), font, 1, (0, 255, 0),2, cv2.LINE_AA)
            cv2.putText(info, '   Action : {}'.format(Action), (270, 30), font, 1, (0, 255,0),2, cv2.LINE_AA)
            
            # Display Apple
            cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]), (self.apple_position[0] + OBJ_SİZE, self.apple_position[1] + OBJ_SİZE),
                          (246, 160, 180), -1)

            # Display Snake
            for position in self.snake_position:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(self.img, (position[0], position[1]), (position[0] + OBJ_SİZE, position[1] + OBJ_SİZE), (0, 255, 0), -1)
                cv2.rectangle(self.img, (position[0], position[1]), (position[0] + OBJ_SİZE, position[1] + OBJ_SİZE), (0, 0, 0), 2)
                
            img_info=np.concatenate([info,self.img])
            cv2.imshow('my_denek', img_info)
            cv2.waitKey(100)
            


        
        

       


        # Change the head position based on the button direction
        if Action == 1:
            self.snake_head[0] += OBJ_SİZE
        elif Action == 0:
            self.snake_head[0] -= OBJ_SİZE
        elif Action == 2:
            self.snake_head[1] += OBJ_SİZE
        elif Action == 3:
            self.snake_head[1] -= OBJ_SİZE


        
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = self.collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
            
            

            self.pased=True

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()


        # On collision kill the snake and print the score
        if self.collision_with_boundaries(self.snake_head) == 1 or self.collision_with_self(self.snake_position) == 1:
            self.done=True



        # ÖDÜL VEYA CEVA DEĞERLERİ
        Sx = self.snake_head[0]
        Sy = self.snake_head[1]
        Deltax = self.apple_position[0]-Sx
        Deltay = self.apple_position[1]-Sy
        S_leng = len(self.snake_position) 
        new_calculation=self.calculate_distance(self.snake_head, self.apple_position)



        """
        # ÖDÜLLER
        if self.past_calculation>new_calculation:  # yakınlasırsa
            step_reward+=1
            self.past_calculation="""

        if self.pased:                             # elma yerse
            step_reward=50
        
        """
        # CEZALAR 
        if self.past_calculation<new_calculation:  # yakınlasırsa
            step_reward-=1
            self.past_calculation=new_calculation
        

        if (self.max_frame+5)<self.step_frame and self.pased: 
            step_reward-=5
        """
        if self.done:
            reward=-10
        



        """step_total=((S_leng*10)+step_reward)/100"""

        step_total = (self.sigmoid(new_calculation)*10+self.sigmoid(self.step_frame)*10 + step_reward)/100

        
        reward=step_total





        """
        # ÖDÜLLER
        if reward>self.prev_reward:  # daha iyi
            reward+=2
            self.prev_reward=reward

        elif reward<self.prev_reward:  # daha kotü
            reward-=2
            self.prev_reward=reward




        print(" P : ",P)   self.prev_reward=10  gecikme=
        print(" SL : ",SL)
        print(" T : ",T)
        print(" NR : ",NR)
        print(" reward : ",reward)"""

        
        img=self.make_mask(ARENA_MASK_size,self.snake_position,self.apple_position,show=False)

        squart=self.get_squart(img,self.snake_head,(80,80)).reshape(-1).reshape(80,80)

        #for s,i in enumerate(squart):
        #    print(i)
        cv2.imshow("Gorus_Acisi",squart)
        cv2.waitKey(1)
        #time.sleep(1)
        #input("devam için enterlayın")

        


        self.step_frame+=1

        if self.pased:                             # elma yerse
            self.pased=False
            self.step_frame=0
            self.past_calculation=new_calculation
            self.max_frame=ARENA_size[0]
            self.prev_reward=0
            self.prev_kazanc=0
        
            
         

        obs = np.array([Sx,Sy,Deltax,Deltay,S_leng,self.step_frame,new_calculation],dtype=np.float32)

        self.observation =np.append(obs,squart)#[-1 for i in range(SNAKE_LEN_GOAL)]a
        #print(" info size : ",deneme.shape)
        #print(deneme)
        


        return  self.observation,np.array(reward,dtype=np.float32),np.array(self.done,dtype=np.float32)





    def reset(self):
        self.done=False

        self.snake_head = [int(ARENA_size[0]/2),int(ARENA_size[1]/2)]
        self.snake_position = [[int(ARENA_size[0]/2),int(ARENA_size[1]/2)],[self.snake_head[0]-OBJ_SİZE,self.snake_head[1]],[self.snake_head[0]-(OBJ_SİZE*2),self.snake_head[1]]]
        self.apple_position = [int(random.randrange(1,ARENA_size[0]/10)*10),int(random.randrange(1,ARENA_size[1]/10)*10)]

        self.score = 0
        self.prev_reward=0
        self.past_calculation=self.calculate_distance(self.snake_head,self.apple_position)
        self.step_frame=0
        self.max_frame=ARENA_size[0]
        self.pased=False
        self.prev_kazanc=0


        # OBS degerleri
        Sx=self.snake_head[0]
        Sy = self.snake_head[1]
        Deltax = self.apple_position[0]-Sx
        Deltay = self.apple_position[1]-Sy
        S_leng=len(self.snake_position)

        

        img=self.make_mask(ARENA_MASK_size,self.snake_position,self.apple_position,show=False)
        squart=self.get_squart(img,self.snake_head,(80,80)).reshape(-1)


        obs = np.array([Sx,Sy,Deltax,Deltay,S_leng,self.step_frame,self.past_calculation],dtype=np.float32)
        self.observation =np.append(obs,squart)#[-1 for i in range(SNAKE_LEN_GOAL)]a


        return np.array(self.observation,dtype=np.float32)

    def get_squart_v2(self,img,show=False):
        
        img =np.array(img,dtype="float32")
        img[img[:,:]==255] = 1

        img[img[:,:]==100] = 1

        img[img[:,:]==0] = 0

        mask=np.zeros((img.shape[0]+2,img.shape[1]+2,1))
        mask[:,:]= 1

        mask[1:-1,1:-1]=img



        if show:
            cv2.rectangle(img, (start_point[0] - size[0], start_point[1] - size[0]),
                               (start_point[0] + size[0], start_point[1] + size[0]),
                          255, 1)


        return mask.astype(np.float32)

    def make_mask(self,size,obje_array,target,show=False,resize=False):
        img = np.zeros(size, dtype='uint8')
        # Display Snaexit
        cv2.rectangle(img, (target[0], target[1]), (target[0] + OBJ_SİZE, target[1] + OBJ_SİZE),255, -1)

        for position in obje_array:
            position=np.array(position,dtype=np.int32)
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.rectangle(img, (position[0], position[1]), (position[0] + OBJ_SİZE, position[1] + OBJ_SİZE),100, -1)

        if resize: img=cv2.resize(img,SİZE).reshape(SİZE[0],SİZE[1],1)

        if show: 
            cv2.imshow("gozlemci_haritası",img)

        return img

    def sigmoid(self,x):
        """
        Sigmoid fonksiyonu hesaplar.

        Parameters:
        x (numpy.ndarray): Giriş verisi veya dizi

        Returns:
        numpy.ndarray: Sigmoid fonksiyonu uygulanmış çıkış
        """
        return 1 / (1 + x)

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
        apple_position = [random.randrange(1,int(ARENA_size[0]/OBJ_SİZE))*10,random.randrange(1,int(ARENA_size[0]/OBJ_SİZE))*10]
        score += 1
        return apple_position, score

    def collision_with_boundaries(self,snake_head):
        if snake_head[0]>=ARENA_size[0] or snake_head[0]<0 or snake_head[1]>=ARENA_size[1] or snake_head[1]<0 : #duvara çarparsa
            return 1
        else:
            return 0

    def collision_with_self(self,snake_position):   # kuyruğuna çarparsa
        snake_head = snake_position[0]
        if snake_head in snake_position[1:]:
            return 1
        else:
            return 0

    def get_squart(self,img,start_point,size,show=False,resize=False):

        size=(np.array(size)/2).astype(np.int32)

        y1=start_point[1] - size[1]
        y2=start_point[1] + size[1]
        x1=start_point[0] - size[0]
        x2=start_point[0] + size[0]

        iw,ih,_=img.shape
        x1_c,x2_c,y1_c,y2_c=0,size[0]*2,0,size[0]*2
        if x1<0:
            x1_c=x1
            x1=0
        if y1<0:
            y1_c=y1
            y1=0
        if x2>iw:
            x2_c=(size[0]*2)-(x2-iw)
        if y2>ih:
            y2_c=(size[1]*2)-(y2-ih)

        squart=img[y1:y2,
                   x1:x2]


        squart =np.array(squart,dtype="float32")
        squart[squart[:,:]==255] = 1

        squart[squart[:,:]==100] = -1

        squart[squart[:,:]==0] = 0




        mask=np.zeros((size[0]*2,size[0]*2,1))
        mask[:,:]=-1

        mw,mh,_=mask.shape
        """
        print("mask_shape : ",mask.shape)
                                print("Squar_shape : ",squart.shape)
                        
        print("y : ",y1,y2)
        print("x : ",x1,x2)
        print("artıklar x1,x2,y1,y2 : ",x1_c,x2_c,y1_c,y2_c)
        print("kare_dagılımı : ",abs(min(y1_c,0)),":",abs(min(mh,y2_c)),abs(min(x1_c,0)),":",abs(min(mw,x2_c)))
                        """
        mask[abs(min(y1_c,0)):abs(min(mh,y2_c)),abs(min(x1_c,0)):abs(min(mw,x2_c))]=squart


        
        if show:
            cv2.rectangle(img, (start_point[0] - size[0], start_point[1] - size[0]),
                               (start_point[0] + size[0], start_point[1] + size[0]),
                          255, 1)


        if resize: 
            mask=cv2.resize(mask,(10,10)).reshape(10,10,1)


        return mask.astype(np.float32)

