import gym
from stable_baselines3 import PPO
from snake_env import snake_game
import os,time
from tqdm import tqdm
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MODEL():

    def __init__(self,model_dir=".",log_dir="."):
        self.model_dir=model_dir
        self.log_dir=log_dir

    def make_file(self):
        print(" dosya name : ",self.model_dir)
        if not os.path.exists(self.model_dir):os.makedirs(self.model_dir)
        if not os.path.exists(self.log_dir):os.makedirs(self.log_dir)

    def make_env(self,render_mode="array"):
        env=snake_game(render_mode=render_mode)
        print("obs : ",env.observation_space)
        env.reset()
        return env

    def make_model(self,env,policy="MlpPolicy",verbose=0,
                    learning_rate = 0.0000321,
                    ent_coef = 0.001,
                    n_steps = 128,
                    batch_size = 128,
                    gamma = 0.99,
                    gae_lambda = 0.95):

        return PPO(policy,env,verbose=verbose,tensorboard_log=self.log_dir,
            device="cuda",learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            n_steps=n_steps,
            gae_lambda=gae_lambda)

    def train(self,model,steps=1000):
            model.learn(total_timesteps=steps,reset_num_timesteps=False,tb_log_name="PPO")
            
        

    def load(self,model,env,path="models/1702159037/4000.zip"):
        return model.load(path,env)

    def load2(self,model,path="models/1702159037/4000.zip"):
        return model.set_parameters(path)
        
    def deneme(self,*args):
        print("="*100)
        print()
        print(" "*30,"CALLBACK")
        print()
        print("="*100)



        print(args)



    def pred(self,env,model,game_size=1,spead_game=100,show=False):
        toplam_ödül=0
        info={"reward":[],
              "step":[]}


        if show:env.render_mode="human"


        for  i in range(game_size):
            done=False
            obs =env.reset()

            ödül=0
            kare=0

            while not done:

                action,_=model.predict(obs)
                obs,reward,done=env.step(action)

               
                #print("obs : ", obs)
                #print("done : ", done)
                if spead_game:time.sleep(spead_game)
                kare+=1
                ödül+=reward

            info["reward"].append(ödül)
            info["step"].append(kare)
            toplam_ödül+=ödül




        if show:env.render_mode="array"

        max_reward=info["reward"][np.argmax(info["reward"])]
        min_reward=info["reward"][np.argmin(info["reward"])]

        max_step=info["step"][np.argmax(info["step"])]
        min_step=info["step"][np.argmin(info["step"])]


        mode_rewad=self.mode(info["reward"])
        mode_step=self.mode(info["step"])

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



        info_tqdm=f"mxr : {str(max_reward)[:5]} , mnr : {str(min_reward)[:5]} , mor : {(str(mode_rewad[0])[:5],str(mode_rewad[1])[:5])} , mxs : {str(max_step)[:3]} , mns : {str(min_step)[:3]} , mos : {(str(mode_step[0])[:3],str(mode_step[1])[:3])} , mer : {str(mean_reward)[:5]} , mes : {str(mean_step)[:3]}"


        return info_tqdm

    def mode(self,array):
        (_, idx, counts) = np.unique(array, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        return array[index],counts[np.argmax(counts)]


#####################################################    MODEL
model_steps=10000
steps=500
name=f"{int(time.time())}"
model=MODEL(model_dir=f"models/{name}",log_dir=f"logs/{name}")

model.make_file()
env=model.make_env()
ppo=model.make_model(env,learning_rate=0.000321,gamma=0.97,n_steps = 64,
                    batch_size = 64,)

ppo=model.load(ppo,env,"models/1703223033/2990000.zip")

#####################################################    TRAİN
progrestbar=tqdm(total=steps,ncols=170)
for i in range(steps):
    model.train(ppo,steps=model_steps)
    ppo.save(f"{model.model_dir}/{model_steps*i}")

    info_tqdm=model.pred(env,ppo,game_size=100,spead_game=False)

    progrestbar.postfix=info_tqdm
    progrestbar.update(1)

env.close()





