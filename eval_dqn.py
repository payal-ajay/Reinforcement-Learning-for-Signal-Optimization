# eval_dqn.py
import os, sys, time
if 'SUMO_HOME' not in os.environ:
    raise RuntimeError("Please set SUMO_HOME environment variable.")
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

import traci
import numpy as np
from stable_baselines3 import DQN
from sumolib import checkBinary

SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "simple_intersection.sumocfg"
MODEL_FILE = "dqn_sumo_tls.zip"

def evaluate(model_path=MODEL_FILE, steps=500):
    model = DQN.load(model_path)
    traci.start([checkBinary(SUMO_BINARY), "-c", SUMO_CONFIG])
    tls = traci.trafficlight.getIDList()[0]
    total_wait = 0.0
    total_halts = 0
    prev_arrived = traci.simulation.getArrivedNumber()

    for step in range(steps):
        # build observation
        lanes = traci.trafficlight.getControlledLanes(tls)
        obs = np.array([traci.lane.getLastStepVehicleNumber(l) for l in lanes], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        phase_index = 0 if int(action) == 0 else 2
        traci.trafficlight.setPhase(tls, phase_index)
        traci.simulationStep()

        for vid in traci.vehicle.getIDList():
            total_wait += traci.vehicle.getWaitingTime(vid)
        for lane in lanes:
            total_halts += traci.lane.getLastStepHaltingNumber(lane)

    arrived = traci.simulation.getArrivedNumber()
    throughput = arrived - prev_arrived
    traci.close()
    print("✅ RL eval finished")
    print(f"Total waiting time: {total_wait:.2f}, total halts: {total_halts}, throughput: {throughput}")
    return {"total_wait": total_wait, "total_halts": total_halts, "throughput": throughput}

if __name__ == "__main__":
    evaluate()
