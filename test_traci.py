import os, sys, time

if 'SUMO_HOME' not in os.environ:
    raise RuntimeError("Please set SUMO_HOME environment variable to your SUMO installation path.")
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

from sumolib import checkBinary
import traci

def main(gui=False, steps=20):
    sumo_bin = checkBinary('sumo-gui' if gui else 'sumo')
    print("DEBUG: About to start SUMO...")
    traci.start([sumo_bin, "-c", "simple_intersection.sumocfg", "--no-step-log", "true"])
    print("✅ SUMO started via TraCI (gui=%s)" % gui)

    try:
        for step in range(steps):
            traci.simulationStep()
            vehs = traci.vehicle.getIDList()
            print(f"Step {step:02d}: vehicles in sim = {len(vehs)} | IDs = {vehs}")
            time.sleep(0.05)
    finally:
        traci.close()
        print("✅ SUMO closed cleanly")

if __name__ == "__main__":
    print("DEBUG: Running main() now...")
    main(gui=True, steps=100)
