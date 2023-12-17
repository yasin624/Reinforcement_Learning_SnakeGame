import gym
from stable_baselines3 import PPO
from snake_env import snake_game
import os,time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

name=int(time.time())
model_dir=f"models/{name}"
log_dir =f"logs/{name}"


if not os.path.exists(model_dir):os.makedirs(model_dir)
if not os.path.exists(log_dir):os.makedirs(log_dir)


TİMESTEP=1000
env=snake_game(progrest_total=int(TİMESTEP/2),render_model="array")

env.reset()


#model=PPO("MlpPolicy",env,verbose=1,tensorboard_log=log_dir,device="cuda", learning_rate=0.0003 )








#model.load("models/1702159037/4000.zip")



# Eğitim parametreleri
total_timesteps = 10000
learning_rate = 0.00321
ent_coef = 0.01
n_steps = 2048
batch_size = 64
gamma = 0.99
gae_lambda = 0.95

# PPO eğitimini başlat
model = PPO("MlpPolicy",env,verbose=1,tensorboard_log=log_dir,device="cuda",learning_rate=learning_rate, ent_coef=ent_coef,
            n_steps=n_steps, batch_size=batch_size, gamma=gamma, gae_lambda=gae_lambda)

for i in range(1,100):

    model.learn(total_timesteps=TİMESTEP,reset_num_timesteps=False,tb_log_name="PPO")
    model.save(f"{model_dir}/{TİMESTEP*i}")
    env.progrest.reset()


env.close()





"""for i in range(1,100):

    model.learn(total_timesteps=TİMESTEP,reset_num_timesteps=False,tb_log_name="PPO")
    model.save(f"{model_dir}/{TİMESTEP*i}")
    env.progrest.reset()

env.close()"""
