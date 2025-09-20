import torch
import traci
from sumo_env import SumoIntersectionEnv
from train_dqn import DQN

def evaluate_dqn(model_path="dqn_model.pth", episodes=5):
    env = SumoIntersectionEnv(max_steps=300)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_waits = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_waiting_time = 0
        total_vehicles = 0

        while not done:
            q_values = model(torch.FloatTensor(state).unsqueeze(0))
            action = torch.argmax(q_values).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # collect metrics
            for edge in traci.edge.getIDList():
                if edge.startswith(":"):
                    continue
                total_waiting_time += traci.edge.getWaitingTime(edge)
                total_vehicles += traci.edge.getLastStepVehicleNumber(edge)

        avg_wait = total_waiting_time / max(1, total_vehicles)
        all_waits.append(avg_wait)

    env.close()
    return sum(all_waits) / len(all_waits)
