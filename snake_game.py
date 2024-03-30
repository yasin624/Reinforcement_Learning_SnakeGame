
import numpy as np
import cv2
import random

class snake_game:
    __Arene_Size = (500, 500, 3)
    __OBJ_SİZE = 10
    done = False

    def __init__(self,spead=1):

        self.render_mode = "human"
        self.reset()
        self.__spead=int(1000/spead)
        self.Action=3

    def Move(self):
        # Change the head position based on the button direction
        if self.Action == 1:
            self.snake_head[0] += self.__OBJ_SİZE
        elif self.Action == 0:
            self.snake_head[0] -= self.__OBJ_SİZE
        elif self.Action == 2:
            self.snake_head[1] += self.__OBJ_SİZE
        elif self.Action == 3:
            self.snake_head[1] -= self.__OBJ_SİZE

        self.snake_position.insert(0, list(self.snake_head))
        self.snake_position.pop()
    def collision_with_apple(self):
        if self.snake_head == self.apple_position:
            self.apple_position= [random.randrange(1, int(self.__Arene_Size[0] / self.__OBJ_SİZE)) * 10,
                                                  random.randrange(1, int(self.__Arene_Size[0] / self.__OBJ_SİZE)) * 10]
            self.snake_position.insert(0, list(self.snake_head))
            self.score+=1
            self.pased = True
            return True

    def collision_with_boundaries(self):
        if self.snake_head[0] >= self.__Arene_Size[0] or self.snake_head[0] < 0 or self.snake_head[1] >= self.__Arene_Size[1] or self.snake_head[
            1] < 0:  # duvara çarparsa
            self.done = True
    def collision_with_self(self):  # kuyruğuna çarparsa
        if self.snake_head in self.snake_position[1:]:
            self.done = True

    def drawing_text(self,img,text,localation,font=cv2.FONT_HERSHEY_SIMPLEX,size=1,color=(0, 255, 0),bold=2,type=cv2.LINE_AA):

        # drawing text ower image
        cv2.putText(img,text, localation, font, size, color, bold, type)
    def render(self,show=True):
        if self.render_mode == "human":
            self.img = np.zeros(self.__Arene_Size, dtype='uint8')
            font = cv2.FONT_HERSHEY_SIMPLEX

            # drawing text ower image
            info = np.zeros((50, self.img.shape[1], self.img.shape[2]))
            info[:, :] = [54, 54, 54]

            self.drawing_text(img=info,text='ScoreQ : {}  |'.format(self.score),localation=(50, 30))
            self.drawing_text(img=info, text='   Action : {}'.format(self.Action), localation= (270, 30))


            # Display Apple
            cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                          (self.apple_position[0] + self.__OBJ_SİZE, self.apple_position[1] + self.__OBJ_SİZE),
                          (246, 160, 180), -1)

            # Display Snake
            for position in self.snake_position:
                cv2.rectangle(self.img, (position[0], position[1]), (position[0] + self.__OBJ_SİZE, position[1] + self.__OBJ_SİZE),
                              (0, 255, 0), -1)
                cv2.rectangle(self.img, (position[0], position[1]), (position[0] + self.__OBJ_SİZE, position[1] + self.__OBJ_SİZE),
                              (0, 0, 0), 2)

            # display snake_head
            cv2.rectangle(self.img, (self.snake_head[0], self.snake_head[1]),
                          (self.snake_head[0] + self.__OBJ_SİZE, self.snake_head[1] + self.__OBJ_SİZE), (0, 0, 255), -1)

            img_info = np.concatenate([info, self.img])

            if show:
                cv2.imshow('SNAKE GAME', img_info)
                cv2.waitKey(int(1000/self.__spead))
            else:
                return img_info
    def step(self, Action):
        self.Action=Action
       # move nex step
        self.Move()
        # On collision kill the snake and print the score
        self.collision_with_boundaries()
        self.collision_with_self()
       # Increase Snake length on eating apple
        self.collision_with_apple()
        self.step_frame += 1

    def reset_pased(self):
        self.pased = False
        self.step_frame = 0

        self.max_frame = self.__Arene_Size[0]
        self.prev_reward = 0
        self.prev_kazanc = 0
    def reset(self):
        self.done = False
        self.score = 0
        self.Action =3
        self.prev_reward = 0
        self.step_frame = 0
        self.max_frame = self.__Arene_Size[0]
        self.pased = False
        self.prev_kazanc = 0

        self.snake_head = [int(self.__Arene_Size[0] / 2), int(self.__Arene_Size[1] / 2)]
        self.snake_position = [[int(self.__Arene_Size[0] / 2), int(self.__Arene_Size[1] / 2)],
                               [self.snake_head[0] - self.__OBJ_SİZE, self.snake_head[1]],
                               [self.snake_head[0] - (self.__OBJ_SİZE * 2), self.snake_head[1]]]
        self.apple_position = [int(random.randrange(1, self.__Arene_Size[0] / 10) * 10),
                               int(random.randrange(1, self.__Arene_Size[1] / 10) * 10)]

        self.total_aple = (self.__Arene_Size[0]*self.__Arene_Size[0]) - len(self.snake_position)



