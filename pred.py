import gym
from stable_baselines3 import PPO
from snake_env import snake_game
import os,time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"



TİMESTEP=1000
env=snake_game(progrest_total=TİMESTEP,render_model="human")
model=PPO("MlpPolicy",env,verbose=1,device="cuda")
model.load("models/1702201041/48000.zip")

episodes=10

for  i in range(episodes):

    done=False

    obs =env.reset()
    odül=0
    while not done:
        action,_=model.predict(obs)

        print("action : ",action)

        obs,reward,done=env.step(action)
        odül+=reward

       
        #print("obs : ", obs)
        #print("done : ", done)
        #time.sleep(1)
    print("odül : ",odül)
