import torch
import traci
from sumo_env import SumoIntersectionEnv
from train_dqn import DQN, ReplayBuffer  # import your DQN class

def test_dqn(model_path="dqn_model.pth", episodes=3):
    env = SumoIntersectionEnv(max_steps=300)

    # Load trained model
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for ep in range(episodes):
        print(f"🚦 Testing Episode {ep+1}")
        state, _ = env.reset()

        done = False
        while not done:
            # Pick best action
            q_values = model(torch.FloatTensor(state).unsqueeze(0))
            action = torch.argmax(q_values).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.close()

if __name__ == "__main__":
    test_dqn()
