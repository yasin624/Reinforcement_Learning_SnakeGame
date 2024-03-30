from  snake_game import snake_game
import keyboard as key
import cv2
import numpy as np


class Play:
    def __init__(self):
        self.Game=snake_game(spead=10)
        self.Action=3
        self.menu="main"
    def Menu(self,menu="main"):

        if menu=="main":
            text="""
        
        START GAME 

[1]   Play New Game 
[2]   countion Game

[esc] Quit 


        WRITER : yalcinyazilimcilik
            
            """
        elif menu == "stopped" and self.Game.done:
            text = """


            !!!!  YOU WİN  !!!!!!

[1]   Play New Game 

[esc] Quit 

            WRITER : yalcinyazilimcilik
                                """

        elif menu == "stopped" and self.Game.done:
            text = """


            GAME OWER 
             YOU LOST

[1]   Play New Game 

[esc] Quit 
                
            WRITER : yalcinyazilimcilik
                        """


        elif menu == "stopped" and not self.Game.done:
            text = """
                
            GAME IS STOPPED 
                
[1]   Play New Game 
[2]   countion Game

[esc] Quit 


            WRITER : yalcinyazilimcilik
                        """
            pass


        return text



    def run(self):
        while True:
            if (key.is_pressed('a') or key.is_pressed("left")) and self.Action != 0 and self.Action != 1:
                self.Action = 0

            elif (key.is_pressed("w") or key.is_pressed("up")) and self.Action != 3 and self.Action != 2:
                self.Action = 3

            elif (key.is_pressed("d") or key.is_pressed("right")) and self.Action != 1 and self.Action != 0:
                self.Action = 1

            elif (key.is_pressed("s") or key.is_pressed("down")) and self.Action != 2 and self.Action != 3:
                self.Action = 2

            elif key.is_pressed("esc"):
                self.menu="stopped"
                break

            if self.Game.score == self.Game.total_aple:
                break
            if self.Game.done:
                break


            self.Game.step(self.Action)
            img = self.Game.render(show=False)
            self.render(img)


    def main(self):
        while True:
            img = self.menu_img(self.Game.render(show=False),menu=self.menu)
            self.render(img)

            if key.is_pressed('1') :
                self.Game.reset()
                self.Action=3

                self.run()


            elif key.is_pressed('2') and not self.Game.done:
                self.run()

            elif key.is_pressed('esc'):
                break

    def menu_img(self,img,menu="stopped"):
        mask=np.zeros(self.Game._snake_game__Arene_Size)
        w,h,z=mask.shape
        for s,T in enumerate(self.Menu(menu).split("\n")):
            self.Game.drawing_text(img=mask,text=T,localation=(100,50+(s*30)),bold=1,size=1/2)


        img[50:w+50,0:h,:]=mask
        return img
    def render(self,img):


        cv2.imshow('SNAKE GAME',img)
        cv2.waitKey(self.Game._snake_game__spead)




if __name__=="__main__":
    env=Play()
    env.main()