import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from sumo_env import SumoIntersectionEnv


# Simple Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.layers(x)


def train_dqn(episodes=5, headless=True):  # only 5 episodes for quick test
    env = SumoIntersectionEnv(max_steps=200)  # shorter simulation
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    memory = deque(maxlen=2000)
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(env.max_steps):
            if np.random.rand() <= epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state))
                    action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])

            memory.append((state, action, reward, next_state, terminated))
            state = next_state

            if terminated:
                break

            # Train DQN
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                states = torch.FloatTensor(np.vstack([m[0] for m in minibatch]))
                actions = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1)
                rewards = torch.FloatTensor([m[2] for m in minibatch])
                next_states = torch.FloatTensor(np.vstack([m[3] for m in minibatch]))
                dones = torch.FloatTensor([float(m[4]) for m in minibatch])

                q_values = model(states).gather(1, actions).squeeze()
                next_q_values = model(next_states).max(1)[0].detach()
                target = rewards + (gamma * next_q_values * (1 - dones))

                loss = criterion(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

    torch.save(model.state_dict(), "dqn_model.pth")
    print("✅ Training finished, model saved as dqn_model.pth")
    env.close()


if __name__ == "__main__":
    train_dqn(episodes=5, headless=True)
