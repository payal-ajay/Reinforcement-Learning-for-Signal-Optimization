from baseline_fixed import run_baseline
from sumo_env import SumoIntersectionEnv
from train_dqn import DQN
import torch
import numpy as np


def evaluate_dqn(model_path, episodes=3, gui=False, max_steps=200):
    """Evaluate a trained DQN agent"""
    env = SumoIntersectionEnv(max_steps=max_steps)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_wait = 0
    total_vehicles = 0

    for ep in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state))
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = np.reshape(next_state, [1, state_size])
            done = terminated or truncated

            total_wait += -reward  # reward = -queue length
            total_vehicles += len(env.vehicles) if hasattr(env, "vehicles") else 1

    env.close()
    avg_wait = total_wait / episodes
    return avg_wait, total_vehicles


if __name__ == "__main__":
    print(" Running baseline fixed-time controller...")
    baseline_wait = run_baseline(episodes=3, max_steps=200)
    print(f" Fixed-time avg wait: {baseline_wait:.2f}")

    print(" Evaluating DQN model...")
    dqn_wait, vehicles = evaluate_dqn("dqn_model.pth", episodes=3, gui=False, max_steps=200)
    print(f" DQN avg wait: {dqn_wait:.2f}")

    # Efficiency calculation
    improvement = ((baseline_wait - dqn_wait) / baseline_wait) * 100 if baseline_wait > 0 else 0
    print(f" Efficiency Improvement: {improvement:.2f}%")

    if dqn_wait < baseline_wait:
        print(" DQN performs better than baseline!")
    else:
        print(" Baseline performed better in this test.")

