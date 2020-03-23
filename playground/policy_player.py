import numpy as np
import time
from pybullet_envs.hexapod.env import hexapod_gym_env
import gym

def main():
  gym_env = hexapod_gym_env.HexapodGymEnv(pd_control_enabled = True,
                                          accurate_motor_model_enabled = True,
                                          leg_model_enabled = True,
                                          motor_overheat_protection = True,
                                          render = True,
                                          on_rack = True,
                                          log_path = "/home/vaibhav/tensorflow1_gpu/lib/python3.5/site-packages/pybullet_envs/hexapod/playground/test1/logs")
  env = gym_env
  sum_reward = 0
  observation = env.reset()
  n = 0
  Action = np.identity(18)*0.25
  while True:
    #action = (np.random.random(18)*2 - 1)/4  #clip random numbers from -0.25 to +0.25 (PI will multiply later on)
    #action = Action[n%18]
    action = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    observation, reward, done, _ = env.step(action)
    print("Observation:",observation[:18])
    #print("Reward",reward)
    #print(done)
    time.sleep(5)
    sum_reward+=reward
    n += 1
    break
    

if __name__ == '__main__':
  main()


