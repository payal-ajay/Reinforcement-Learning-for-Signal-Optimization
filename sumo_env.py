import os
import traci
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sumolib import checkBinary


class SumoIntersectionEnv(gym.Env):
    def __init__(self, sumo_cfg="simple_intersection.sumocfg", max_steps=200):
        super(SumoIntersectionEnv, self).__init__()
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.step_count = 0

        # Connect to SUMO
        if traci.isLoaded():
            traci.close()
        traci.start([checkBinary("sumo"), "-c", self.sumo_cfg])

        # Detect available TLS automatically
        tls_ids = traci.trafficlight.getIDList()
        print("✅ Available TLS IDs:", tls_ids)
        if not tls_ids:
            raise RuntimeError("No traffic lights found in the network.")
        self.traffic_light_id = tls_ids[0]

        # Get phases dynamically
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.traffic_light_id)[0]
        self.phases = [phase.state for phase in logic.getPhases()]
        print(f"✅ Phases for {self.traffic_light_id}: {self.phases}")

        # Gym action/observation spaces
        self.action_space = spaces.Discrete(len(self.phases))  # pick a phase index
        self.observation_space = spaces.Box(low=0, high=1000, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if traci.isLoaded():
            traci.close()
        traci.start([checkBinary("sumo"), "-c", self.sumo_cfg])
        self.step_count = 0
        return self._get_state(), {}

    def step(self, action):
        # Set traffic light phase
        traci.trafficlight.setRedYellowGreenState(self.traffic_light_id, self.phases[action])

        traci.simulationStep()
        self.step_count += 1

        state = self._get_state()
        reward = -sum(state)  # minimize total queue length
        terminated = self.step_count >= self.max_steps
        truncated = False
        return state, reward, terminated, truncated, {}

    def _get_state(self):
        lanes = traci.trafficlight.getControlledLanes(self.traffic_light_id)
        counts = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes[:4]]
        return np.array(counts, dtype=np.float32)

    def close(self):
        if traci.isLoaded():
            traci.close()
