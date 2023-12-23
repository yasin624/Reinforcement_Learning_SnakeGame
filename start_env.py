from snake_env import snake_game
import numpy as np

import time
env =snake_game(render_mode="human")



def mode(array):
    (_, idx, counts) = np.unique(array, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]
    return array[index],counts[np.argmax(counts)]

episodes=50

toplam_ödül=0
info={"reward":[],
      "step":[]}

for  i in range(episodes):
    done=False
    obs =env.reset()
    ödül=0
    kare=0
    while not done:
        random_action=env.action_space.sample()


        obs,reward,done,_=env.step(random_action)

        kare+=1
        ödül+=reward
    info["reward"].append(ödül)
    info["step"].append(kare)

    toplam_ödül+=ödül

    
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