import traci
import sumolib
import numpy as np

def run_baseline(episodes=1, max_steps=200):
    """
    Fixed-time baseline traffic light controller.
    Runs SUMO simulation with predefined signal phases.
    Returns the average waiting time per vehicle.
    """
    total_wait = 0
    for ep in range(episodes):
        sumoBinary = "sumo"  # change to "sumo-gui" if you want GUI
        sumoCmd = [sumoBinary, "-c", "simple_intersection.sumocfg", "--no-step-log", "true"]
        traci.start(sumoCmd)

        tls_id = traci.trafficlight.getIDList()[0]
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        phases = logic.getPhases()

        step = 0
        while step < max_steps:
            phase_index = (step // 30) % len(phases)  # fixed 30s per phase
            traci.trafficlight.setPhase(tls_id, phase_index)
            traci.simulationStep()

            waiting_times = [traci.vehicle.getWaitingTime(vid) for vid in traci.vehicle.getIDList()]
            total_wait += np.mean(waiting_times) if waiting_times else 0

            step += 1

        traci.close()

    avg_wait = total_wait / episodes
    return avg_wait
