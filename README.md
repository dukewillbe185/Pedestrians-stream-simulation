# Simulation-of-pedestrians-stream-when-crossing-roads
This project implements a simulation system for optimizing bidirectional pedestrian flow using Deep Q-Network (DQN) reinforcement learning.
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
## ORCA-DWA Pedestrian Flow Simulation Based on DQN Training
## Table of Contents
1. **Project Overview**
2. **Environment Requirements**
3. **Installation Steps**
4. **Usage Instructions**
5. **Code Structure**
6. **Custom Settings**
7. **Output Results**
8. **Common Issues**
### Project Overview
This simulation system models pedestrian movement in a confined space where pedestrians move in opposite directions. The project aims to optimize pedestrian flow and minimize collisions by using a DQN agent to learn to adjust parameters of the Dynamic Window Approach (DWA) algorithm.
Key features:
Bidirectional pedestrian flow  
Collision avoidance using ORCA (Optimal Reciprocal Collision Avoidance)  
Local path planning using DWA  
DQN learning for optimal DWA parameters  
Pygame-based visualization  
### Environment Requirements
Python 3.9  
PyTorch == 2.4.0  
CUDA == cu118  
Gym == 0.26.1  
Pygame == 2.6.0  
NumPy    
### Installation Steps
1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/pedestrian-flow-simulation.git
   cd pedestrian-flow-simulation
3. **Install required packages**:
   ```bash
   pip install torch gym pygame numpy
   
### Usage Instructions
To run the simulation and train the DQN agent, execute the following command:
1. **Run module 1**:
   ```bash
   python module1.py
   ```
This will start the training process and open a Pygame window displaying the simulation. Training progress will be logged in the pedestrian_flow.log file.
### Code Structure
**PedestrianFlowEnv**: Main environment class implementing the OpenAI Gym interface.  
**Pedestrian**: Class representing individual pedestrians.  
**DQN**: Deep Q-Network model.  
**DQNAgent**: Agent class for learning and decision-making using DQN.  
**ReplayBuffer**: Buffer for storing and sampling training experiences.  
**train_dqn**: Main training loop.  
### Custom Settings
There are many adjustable aspects in the code, such as DQN parameters, number of pedestrians, animation window length and width, etc.  
Adjust `max_pedestrians`, `width`, and `height` parameters in the `PedestrianFlowEnv` constructor.  
Modify the DQN architecture in the `DQN` class.  
Adjust the reward function in the `_calculate_reward` method of `PedestrianFlowEnv`.  
Change training parameters (such as number of episodes, batch size, etc.) in the `train_dqn` function.  
### Output Results
The simulation will be visualized in a Pygame window.  
Training progress is logged in the pedestrian_flow.log file.  
DQN model weights are saved every 100 episodes as dqn_model_episode_X.pth.  
### Common Issues
If you encounter CUDA out of memory errors, try reducing the batch_size or run on CPU by changing device = torch.device("cpu") in the DQNAgent class.  
If visualization issues occur, ensure Pygame is installed correctly and your system supports GUI applications.
