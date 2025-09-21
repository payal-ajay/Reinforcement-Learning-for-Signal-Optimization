DQN for Traffic Signal Control using SUMO
This project demonstrates the use of a Deep Q-Network (DQN) agent to optimize traffic flow at a single intersection simulated in SUMO (Simulation of Urban MObility). The goal is to train an intelligent agent that can learn to manage traffic light phases more efficiently than a static, fixed-time controller, ultimately reducing vehicle waiting times and improving throughput.

Features

Reinforcement Learning Agent: The core of the project is a DQN model built using PyTorch and Stable-Baselines3. The model was trained for 20,000 timesteps on a Windows 11 system with Python 3.12.3.



Custom SUMO Environment: The sumo_env.py script provides a custom Gymnasium environment  that interfaces with the SUMO simulator via TraCI. The observation space is a 16-element array representing vehicle counts in different lanes. The action space has 4 discrete actions, corresponding to different traffic light phases.



Performance Evaluation: The project includes multiple scripts to evaluate the DQN agent's performance. It can be compared to a fixed-time baseline using metrics like total waiting time, average waiting time, and total halts.




Portable Configuration: All simulation settings, including the network topology (.net.xml) and vehicle routes (.rou.xml), are defined in easily modifiable XML files.

Getting Started
Prerequisites
SUMO: Make sure the SUMO simulator is installed and that the SUMO_HOME environment variable is set to your installation path.

Python Libraries: The project's dependencies are listed in requirements.txt. You can install them with the following command:

Bash

pip install -r requirements.txt
Project Structure
**src/**:

train_dqn.py: Script to train a new DQN model.

sumo_env.py: The custom Gymnasium environment for the simulation.

eval.py, eval_dqn.py: Scripts to evaluate the trained DQN model.

compare.py: Compares the DQN agent's performance against the baseline.

fixed_controller.py, baseline_fixed.py: Implementations of the fixed-time traffic light controller.

test_traci.py, check_edges.py: Utility scripts for testing the SUMO connection and network.

**sim_data/**:

simple_intersection.netccfg: Configuration for generating the road network.

simple_intersection.nod.xml, simple_intersection.edg.xml: Defines the nodes and edges for the network.


simple_intersection.rou.xml: Defines vehicle types and traffic flow.



simple_intersection.sumocfg: Main configuration for the simulation run.


**model/**:

dqn_model.pth: The trained PyTorch model.


dqn_sumo_tls.zip: A compressed file containing the model and other training details, including the system info.



How to Run
Run the Comparison:
The main script to see the project in action is compare.py. It runs both the baseline and the DQN model to show the performance difference.

Bash

python compare.py
Train a New Model:
To train a new DQN model, run the train_dqn.py script. The script will save the trained model to a file named dqn_model.pth.

Bash

python train_dqn.py
Individual Simulations:

To run the fixed-time baseline with a GUI: python fixed_controller.py.

To evaluate the DQN model with a GUI: python eval.py
