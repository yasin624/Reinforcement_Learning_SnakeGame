from snake_env import snake_game
import time
env =snake_game()


episodes=50

for  i in range(episodes):

    done=False

    obs =env.reset()

    while not done:
        random_action=env.action_space.sample()

        print(random_action)

        obs,reward,done,info=env.step(random_action)

        print("reward : ",reward)
        print("obs : ", obs)
        print("done : ", done)
        print("info : ", info)

