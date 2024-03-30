
import gym
from gym import spaces
import numpy as np
import cv2
from snake_game import snake_game


print("version 40")

class ENV(gym.Env):

    def __init__(self,render_mode="human",Sk=1.55,Pk=0.1,Tk=0.1,Rk=1.1):
        super(gym.Env,self).__init__()
        
        self.Game=snake_game()

        self.action_space=spaces.Discrete(4)   # seçim yapabilme değeri
        self.observation_space=spaces.Box(low=-self.Game._snake_game__Arene_Size[0],high=self.Game._snake_game__Arene_Size[1],shape=(7+(52*52),),dtype=np.float32)

        self.Game.render_mode=render_mode

        self.ARENA_MASK_size=np.array((500, 500, 1),dtype=np.int32)
    def step(self,Action):
        step_reward=0
        self.Game.render()


        self.Game.Action=Action
        self.Game.Move()

        # Increase Snake length on eating apple
        self.Game.collision_with_apple()

        # On collision kill the snake and print the score
        self.Game.collision_with_boundaries()
        self.Game.collision_with_self()
        


        # ÖDÜL VEYA CEVA DEĞERLERİ
        Sx = self.Game.snake_head[0]
        Sy = self.Game.snake_head[1]
        Deltax = self.Game.apple_position[0]-Sx
        Deltay = self.Game.apple_position[1]-Sy
        S_leng = len(self.Game.snake_position)
        new_calculation=self.calculate_distance(self.Game.snake_head, self.Game.apple_position)


        if self.Game.pased:                             # elma yerse
            step_reward=self.Game.score*10



        """step_total=((S_leng*10)+step_reward)/100"""

        reward=((250-new_calculation)/100+step_reward)/100

        
        if self.Game.done:
            reward=-((500*500)-len(self.Game.snake_position))/1000

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

        
        img=self.make_mask(self.ARENA_MASK_size,self.Game.snake_position,self.Game.apple_position,show=False,resize=(50,50))

        squart=self.get_squart_v2(img).reshape(-1)

        #for s,i in enumerate(squart):
        #    print(i)
        #cv2.imshow("Gorus_Acisi",squart)
        #cv2.waitKey(1)
        #time.sleep(1)
        #input("devam için enterlayın")

        


        self.Game.step_frame+=1

        if self.Game.pased:                             # elma yerse
            self.Game.reset_pased()
            self.past_calculation = new_calculation

        
            
         

        obs = np.array([Sx,Sy,Deltax,Deltay,S_leng,self.Game.step_frame,new_calculation],dtype=np.float32)

        self.observation =np.append(obs,squart)#[-1 for i in range(SNAKE_LEN_GOAL)]a
        #print(" info size : ",deneme.shape)
        #print(deneme)
        


        return  self.observation,np.array(reward,dtype=np.float32),np.array(self.Game.done,dtype=np.float32),{}





    def reset(self):
        self.Game.reset()

        snake_head=self.Game.snake_head
        snake_position=self.Game.snake_position
        apple_position=self.Game.apple_position

        # OBS degerleri
        Sx=snake_head[0]
        Sy = snake_head[1]
        Deltax = apple_position[0]-Sx
        Deltay = apple_position[1]-Sy
        S_leng=len(snake_position)

        self.past_calculation = self.calculate_distance(snake_head, apple_position)
        

        img=self.make_mask(self.ARENA_MASK_size,snake_position,apple_position,show=False,resize=(50,50))
        squart=self.get_squart_v2(img,show=False).reshape(-1)


        obs = np.array([Sx,Sy,Deltax,Deltay,S_leng,self.Game.step_frame,self.past_calculation],dtype=np.float32)
        self.observation =np.append(obs,squart)#[-1 for i in range(SNAKE_LEN_GOAL)]a


        return np.array(self.observation,dtype=np.float32)

    def get_squart_v2(self,img,show=False):
        
        img =np.array(img,dtype="float32")
        img[img[:,:]==255] = 1

        img[img[:,:]==200] = 0.5

        img[img[:,:]==100] = -1

        img[img[:,:]==0] = 0

        mask=np.zeros((img.shape[0]+2,img.shape[1]+2,1))
        mask[:,:]= -1

        mask[1:-1,1:-1]=img



        if show:
            cv2.imshow("deneme",mask)
            cv2.waitKey(200)


        return mask.astype(np.float32)

    def make_mask(self,size,obje_array,target,show=False,resize=False):
        img = np.zeros(size, dtype='uint8')
        # Display Snaexit
        cv2.rectangle(img, (target[0], target[1]), (target[0] + self.Game._snake_game__OBJ_SİZE, target[1] + self.Game._snake_game__OBJ_SİZE),255, -1)

        

        for position in obje_array:
            position=np.array(position,dtype=np.int32)
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.rectangle(img, (position[0], position[1]), (position[0] + self.Game._snake_game__OBJ_SİZE, position[1] + self.Game._snake_game__OBJ_SİZE),100, -1)

        #snake_head
        snake_head=obje_array[0]
        cv2.rectangle(img, (snake_head[0],snake_head[1]), (snake_head[0] + self.Game._snake_game__OBJ_SİZE,snake_head[1] + self.Game._snake_game__OBJ_SİZE), 200, -1)


        if resize: img=cv2.resize(img,resize).reshape(resize[0],resize[1],1)

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

    def get_squart(self,img,start_point,size,show=False,resize=False):

        size=(np.array(size)/2).astype(np.int32)

        y1=start_point[1] - size[1]
        y2=start_point[1] + size[1]
        x1=start_point[0] - size[0]
        x2=start_point[0] + size[0]
        print(" x1,x2,y1,y2 : ",x1,x2,y1,y2)

        iw,ih,_=img.shape
        x1_c,x2_c,y1_c,y2_c=0,size[0]*2,0,size[0]*2

        print("  x1_c,x2_c,y1_c,y2_c : ", x1_c,x2_c,y1_c,y2_c)
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

        print(" x1,x2,y1,y2 : ",x1,x2,y1,y2)
        squart=img[y1:y2,
                   x1:x2]

        print(" squart : ",squart.shape)
        squart =np.array(squart,dtype="float32")
        squart[squart[:,:]==255] = 2
        squart[squart[:,:]==200] = 1

        squart[squart[:,:]==100] = -1
        squart[squart[:,:]==0] = 0
        print(" squart : ",squart.shape)



        mask=np.zeros((size[0]*2,size[0]*2,1))
        mask[:,:]=-1

        mw,mh,_=mask.shape

        print("mask : ",mask.shape," img :",img.shape, " squart : ",squart.shape)
        """
        print("mask_shape : ",mask.shape)
                                print("Squar_shape : ",squart.shape)
                        
        print("y : ",y1,y2)
        print("x : ",x1,x2)
        print("artıklar x1,x2,y1,y2 : ",x1_c,x2_c,y1_c,y2_c)
        print("kare_dagılımı : ",abs(min(y1_c,0)),":",abs(min(mh,y2_c)),abs(min(x1_c,0)),":",abs(min(mw,x2_c)))
                        """
        try:

            mask[abs(min(y1_c,0)):abs(min(mh,y2_c)),abs(min(x1_c,0)):abs(min(mw,x2_c))]=squart
        except:
            mask[:,:]=squart


        if show:
            cv2.rectangle(img, (start_point[0] - size[0], start_point[1] - size[0]),
                               (start_point[0] + size[0], start_point[1] + size[0]),
                          255, 1)


        if resize:
            mask=cv2.resize(mask,(10,10)).reshape(10,10,1)


        return mask.astype(np.float32)

