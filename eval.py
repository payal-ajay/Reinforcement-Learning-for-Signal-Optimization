# eval.py
import numpy as np
from stable_baselines3 import DQN
from sumo_env import SumoIntersectionEnv

def main():
    env = SumoIntersectionEnv(sumo_cfg="simple_intersection.sumocfg", action_duration=5, step_length=1.0, sumo_binary="sumo-gui")
    model = DQN.load("dqn_sumo_tls", env=env)
    n_eps = 3
    for ep in range(n_eps):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            done = terminated or truncated
        print(f"Episode {ep} total reward: {total:.2f}")
    env.close()

if __name__ == "__main__":
    main()
