# fixed_controller.py
import os, sys, time
if 'SUMO_HOME' not in os.environ:
    raise RuntimeError("Please set SUMO_HOME environment variable")
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

from sumolib import checkBinary
import traci

def run(gui=True, steps=200):
    sumo_bin = checkBinary('sumo-gui' if gui else 'sumo')
    traci.start([sumo_bin, "-c", "simple_intersection.sumocfg", "--no-step-log", "true"])
    tls_list = traci.trafficlight.getIDList()
    if not tls_list:
        print("No traffic lights found")
        traci.close()
        return
    tls = tls_list[0]
    print("Controlling TLS:", tls)

    try:
        for step in range(steps):
            # simple 15s cycle: 10 steps NS green, 5 steps yellow, then EW green...
            cycle = step % 30
            if cycle < 10:
                # NS green, EW red — pattern depends on your TLS program; try "GrGr" or "GrGr"
                try:
                    traci.trafficlight.setRedYellowGreenState(tls, "GrGr")
                except Exception:
                    pass
            elif cycle < 12:
                # NS yellow
                try:
                    traci.trafficlight.setRedYellowGreenState(tls, "yryr")
                except Exception:
                    pass
            elif cycle < 22:
                # EW green
                try:
                    traci.trafficlight.setRedYellowGreenState(tls, "rGrG")
                except Exception:
                    pass
            else:
                try:
                    traci.trafficlight.setRedYellowGreenState(tls, "ryry")
                except Exception:
                    pass

            traci.simulationStep()
            time.sleep(0.05)
    finally:
        traci.close()

if __name__ == "__main__":
    run(gui=True, steps=300)
