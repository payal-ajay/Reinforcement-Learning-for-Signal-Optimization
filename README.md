DQN for Traffic Signal Control using SUMO
This project demonstrates the use of a Deep Q-Network (DQN) agent to optimize traffic flow at a single intersection simulated in SUMO (Simulation of Urban MObility). The goal is to train an intelligent agent that can learn to manage traffic light phases more efficiently than a static, fixed-time controller, ultimately reducing vehicle waiting times and improving throughput.

Features

Reinforcement Learning Agent: The core of the project is a DQN model built using PyTorch and Stable-Baselines3. The model was trained for 20,000 timesteps on a Windows 11 system with Python 3.12.3.





Custom SUMO Environment: The sumo_env.py script provides a custom Gymnasium environment that interfaces with the SUMO simulator via TraCI. The observation space is a 16-element array representing vehicle counts in different lanes. The action space has 4 discrete actions, corresponding to different traffic light phases.



Performance Comparison: The compare.py script automatically evaluates the trained DQN model against a traditional fixed-time traffic controller, highlighting the efficiency improvements achieved by the agent.


Reusable Components: The project includes various utility scripts for generating the simulation network, evaluating models, and inspecting the environment.

Getting Started
Prerequisites

SUMO: Make sure the SUMO simulator is installed and that the SUMO_HOME environment variable is set to your installation path.

Python Libraries: All required Python dependencies are listed in requirements.txt. You can install them with a package manager like 

pip.

Project Structure
src/: This folder contains the main Python scripts for the project.


train_dqn.py: The script used to train a new DQN model.


sumo_env.py: The custom Gymnasium environment for the simulation.


eval.py, eval_dqn.py: Scripts for evaluating the trained DQN model.






compare.py: The main script for comparing the DQN agent with the baseline.



fixed_controller.py, baseline_fixed.py: Implementations of the fixed-time traffic light controller.





test_traci.py, check_edges.py: Utility scripts for testing the SUMO connection and network.



sim_data/: This folder contains the SUMO network and route files.


simple_intersection.net.xml, simple_intersection.nod.xml, simple_intersection.edg.xml, simple_intersection.rou.xml: Defines the road network and traffic flows.




simple_intersection.sumocfg: The main configuration file for the simulation.

model/: This folder contains the trained model files.


dqn_model.pth: The saved PyTorch model.


dqn_sumo_tls.zip: A compressed file containing the model and other training details, including the system info.


How to Run
Run the Comparison:
The main purpose of this project is to compare the intelligent agent with a standard controller. You can see the results by running the compare.py script. It will automatically run both the baseline and the DQN model, and then print the average waiting times and efficiency improvements.

Train a New Model:
If you want to train a new model from scratch, you can run the train_dqn.py script. After it finishes, it will save a new model file named dqn_model.pth.

Individual Simulations:
To see the simulation in action, you can run the individual evaluation scripts with the GUI enabled. For example, you can run the fixed_controller.py script to see the baseline in action, or eval_dqn.py to see the trained agent in action.

