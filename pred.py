import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from snake_env import snake_game
from no_death import snake_game as no_death
import torch
import os,time,json
import numpy as np
# Kalitli modeli yükleyin    132 -1 900 adım 1310000.zip,1320000.zip ikisinden biri

#model_dir="models/1702911197/24900000.zip"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#model_dir="models/1703223033/2990000.zip"
#model_dir="models/1703333347/280000.zip"
model_dir="models/1703333347/2130000.zip"



def data_kayıt(data,name):
    with open(name,"w") as file:
        json.dump(data,file)


kayıt_dir="video"
model_name=str(time.time()).split(".")[1]
if not os.path.exists(kayıt_dir):os.makedirs(kayıt_dir)
def mode(array):
    (_, idx, counts) = np.unique(array, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]
    return array[index],counts[np.argmax(counts)]

toplam_ödül=0
info={"reward":[],
      "step":[]}


#env=snake_game(render_mode="human")
env=no_death(render_mode="human")
model=PPO("MlpPolicy",env,verbose=1,)

model = model.load(model_dir)
episodes=20
print("model name : ",model_dir)
time.sleep(1)

metadata={}
for  i in range(episodes):

    done=False
    if not os.path.exists(kayıt_dir+"/"+model_name):os.makedirs(kayıt_dir+"/"+model_name)

    obs =env.reset(kayıt_name=kayıt_dir+"/"+model_name+"/"+str(i)+"_"+model_name)
    ödül=0
    kare=0
    child={}
    while not done:

        action,_=model.predict(obs)
        obs,reward,done=env.step(action)

       
        #print("obs : ", obs)
        #print("done : ", done)
        #time.sleep(0.05)
        kare+=1
        ödül+=reward


    info["reward"].append(ödül)
    info["step"].append(kare)

    toplam_ödül+=ödül


    child["step"]=kare
    child["odül"]=ödül
    child["scor"]=env.score

    metadata[str(i)+"_"+model_name]=child

data_kayıt(metadata,kayıt_dir+"/"+model_name+".json")

max_reward=info["reward"][np.argmax(info["reward"])]
min_reward=info["reward"][np.argmin(info["reward"])]

max_step=info["step"][np.argmax(info["step"])]
min_step=info["step"][np.argmin(info["step"])]


mode_rewad=mode(info["reward"])
mode_step=mode(info["step"])


mean_reward=np.mean(info["reward"])
mean_step=np.mean(info["step"])

info_model=f"""
=====================================

max  reward value   : {max_reward}
min  reward value   : {min_reward}
mode reward value   : {mode_rewad}

max   step  value   : {max_step}
max   step  value   : {min_step}
mode  step  value   : {mode_step}

mean  reward  value   : {mean_reward}
mean  step    value   : {mean_step}
======================================

info  value   : {info}
"""

print(info_model)