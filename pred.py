
from stable_baselines3 import PPO
from snake_env import ENV

import os,time,json
import numpy as np
# Kalitli modeli yükleyin    132 -1 900 adım 1310000.zip,1320000.zip ikisinden biri

def mode(array):
    (_, idx, counts) = np.unique(array, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]
    return array[index],counts[np.argmax(counts)]

def data_kayıt(data,name):
    with open(name,"w") as file:
        json.dump(data,file)


def data_info(info):
    max_reward=info["reward"][np.argmax(info["reward"])]
    min_reward=info["reward"][np.argmin(info["reward"])]

    max_step=info["step"][np.argmax(info["step"])]
    min_step=info["step"][np.argmin(info["step"])]

    max_scor=info["scor"][np.argmax(info["scor"])]
    min_scor=info["scor"][np.argmin(info["scor"])]

    mode_rewad=mode(info["reward"])
    mode_step=mode(info["step"])
    mode_scor=mode(info["scor"])

    mean_reward=np.mean(info["reward"])
    mean_step=np.mean(info["step"])
    mean_scor=np.mean(info["scor"])


    info_model=f"""
            =====================================

            max  reward value   : {max_reward}
            min  reward value   : {min_reward}
            mode reward value   : {mode_rewad}

            max   step  value   : {max_step}
            min   step  value   : {min_step}
            mode  step  value   : {mode_step}

            max   scor  value   : {max_scor}
            min   scor  value   : {min_scor}
            mode  scor  value   : {mode_scor}

            mean  reward  value   : {mean_reward}
            mean  step    value   : {mean_step}
            mean  scor    value   : {mean_scor}
            ======================================

            info  value   : {info}
            """

    print(info_model)


def normal_pred(env,model):
    toplam_ödül = 0
    info = {"reward": [],
            "step": [],
            "scor":[]}

    for i in range(episodes):

        done = False
        if not os.path.exists(kayıt_dir + "/" + model_name): os.makedirs(kayıt_dir + "/" + model_name)

        obs = env.reset()
        ödül = 0
        kare = 0
        child = {}
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done,_ = env.step(action)

            # print("obs : ", obs)
            # print("done : ", done)
            # time.sleep(0.05)
            kare += 1
            ödül += reward

            child["step"] = kare
            child["odül"] = ödül

            metadata[str(i) + "_" + model_name] = child

        info["reward"].append(ödül)
        info["step"].append(kare)
        info["scor"].append(env.Game.score)

        toplam_ödül += ödül

    return toplam_ödül, info



#model_dir="models/1702911197/24900000.zip"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#model_dir="models/1703223033/2990000.zip"
#model_dir="models/1703333347/280000.zip"
model_dir="models/1711694388/40000.zip"


episodes=20
metadata={}
kayıt_dir="video"
model_name=str(time.time()).split(".")[1]



if not os.path.exists(kayıt_dir):os.makedirs(kayıt_dir)


env=ENV(render_mode="human")
model=PPO("MlpPolicy",env,verbose=1,)

model = model.load(model_dir)
episodes=20

print("model name : ",model_dir)


toplam_ödül,info =normal_pred(env,model)

data_kayıt(metadata,kayıt_dir+"/"+model_name+".json")


data_info(info)