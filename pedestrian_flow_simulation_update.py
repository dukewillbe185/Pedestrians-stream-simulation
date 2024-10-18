import gym
from gym import spaces
import numpy as np
import pygame
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
import collections

# DQN模型定义（保持不变）
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
# 经验回放缓冲区（保持不变）
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    

class PedestrianFlowEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, width=800, height=400, max_pedestrians=50):
        super().__init__()
        self.width = width
        self.height = height
        self.max_pedestrians = max_pedestrians
        self.sidewalk_width = 50
        self.render_mode = render_mode
        self.blue_preference = 0.5  # 生成智能体位置偏好
        self.preference_learning_rate = 0.0001  # 更改蓝色智能体偏好学习率
        self.update_interval = 10000  # 参数更新的时间step
        self.steps_since_last_update = 0
        self.total_steps = 0
        self.episode_reward = 0

        self.action_space = spaces.Discrete(11**5)  

        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.max_pedestrians * 5 + 7,),  
            dtype=np.float32
        )

        self.pedestrians = []
        self.completed = 0
        self.steps = 0
        self.dwa_params = {'alpha': 0.5, 'beta': 0.5, 'gamma': 0.5, 'delta': 0.5, 'epsilon': 0.5}

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pedestrian Flow Simulation")
            self.clock = pygame.time.Clock()

    def _get_obs(self):
        obs = []
        for ped in self.pedestrians:
            obs.extend([ped.x / self.width, ped.y / self.height, ped.vx / 3, ped.vy / 3, ped.direction])
        
        obs.extend([0, 0, 0, 0, 0] * (self.max_pedestrians - len(self.pedestrians)))
        
        obs.extend([self.completed / self.max_pedestrians, 
                    len(self.pedestrians) / self.max_pedestrians,
                    self.steps / 1000,
                    self.dwa_params['alpha'],
                    self.dwa_params['beta'],
                    self.dwa_params['gamma'],
                    self.dwa_params['delta']])
        
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        return {
            "completed": self.completed,
            "steps": self.steps,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.blue_preference = 0.5  # 每个epoch开始重置智能体生成位置偏好
        
        self.pedestrians = []
        self.completed = 0
        self.steps = 0

        self.steps_since_last_update = 0
        self.episode_reward = 0

        for _ in range(self.max_pedestrians):
            self._add_pedestrian()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
        
    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        self.steps_since_last_update += 1
        reward = self._calculate_reward()
        self._update_generation_preference(reward)
        
        self._update_dwa_params(action)

        for ped in self.pedestrians[:]:
            old_x, old_y = ped.x, ped.y
            
            nearby_pedestrians = [other for other in self.pedestrians if other != ped and np.linalg.norm(np.array([ped.x - other.x, ped.y - other.y])) < 20]
            if nearby_pedestrians:
                new_velocity = self._apply_orca_dwa(ped, nearby_pedestrians)
                ped.vx, ped.vy = new_velocity
            else:
                ped.vx = ped.speed * ped.direction
                ped.vy = 0
            
            ped.move()
            ped.total_distance += np.sqrt((ped.x - old_x)**2 + (ped.y - old_y)**2)
            ped.time_elapsed += 1

            if (ped.direction == 1 and ped.x > self.width - self.sidewalk_width) or \
            (ped.direction == -1 and ped.x < self.sidewalk_width):
                self.completed += 1
                completed_time = ped.time_elapsed
                completed_distance = ped.total_distance
                logging.info(f"Pedestrian completed: Time = {completed_time}, Distance = {completed_distance}")
                
                self.pedestrians.remove(ped)
                self._add_pedestrian()

        observation = self._get_obs()
        reward = self._calculate_reward()
        self.episode_reward += reward

        if self.steps_since_last_update >= self.update_interval:
            self._update_parameters()
            self.steps_since_last_update = 0

        done = False  
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, False, info
    
    def _update_parameters(self):
        # 更新DWA参数
        for param in self.dwa_params:
            self.dwa_params[param] += np.random.normal(0, 0.1)  # 加入高斯分布噪音
            self.dwa_params[param] = max(0, min(1, self.dwa_params[param]))  
        
        self._update_generation_preference(self.episode_reward / self.update_interval)
        
        # 打印当前状态
        print(f"\nParameters updated at step {self.total_steps}:")
        print(f"DWA parameters: {self.dwa_params}")
        print(f"Blue preference: {self.blue_preference:.4f}")
        print(f"Average reward: {self.episode_reward / self.update_interval:.4f}\n")
        
        # 充值episode奖励
        self.episode_reward = 0

    def _add_pedestrian(self):
        direction = random.choice([-1, 1])
        x = self.width - self.sidewalk_width/2 if direction == -1 else self.sidewalk_width/2
        
        if direction == 1:  
            y = random.uniform(0, self.height) if random.random() > self.blue_preference else random.uniform(0, self.height/2)
        else: 
            y = random.uniform(0, self.height) if random.random() > (1 - self.blue_preference) else random.uniform(self.height/2, self.height)
        
        self.pedestrians.append(Pedestrian(x, y, direction))

    def _update_generation_preference(self, reward):
        # 更新preference参数
        if reward > 0:
            if self.blue_preference > 0.5:
                self.blue_preference += self.preference_learning_rate
            else:
                self.blue_preference -= self.preference_learning_rate
        else:
            if self.blue_preference > 0.5:
                self.blue_preference -= self.preference_learning_rate
            else:
                self.blue_preference += self.preference_learning_rate
        
        self.blue_preference = max(0, min(1, self.blue_preference))

    def _update_dwa_params(self, action):
        param_values = [i/10 for i in range(11)]  # 生成0.0到1.0之间的11个值
        params = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        for i, param in enumerate(params):
            self.dwa_params[param] = param_values[action % 11]
            action //= 11

    def _apply_orca_dwa(self, ped, nearby_pedestrians):
    # 寻找前方的领导者
        leader = self._find_leader(ped)
        ped.following = leader
        
        # 调整速度以跟随领导者
        ped.adjust_speed_to_follow(leader)
        
        # 计算 ORCA 避碰速度
        orca_velocity = self._orca(ped, nearby_pedestrians)
        
        # 计算 DWA 速度
        dwa_velocity = self._dwa(ped, orca_velocity)
        
        # 计算跟随向量
        if leader:
            follow_vector = np.array([leader.x - ped.x, leader.y - ped.y])
            distance = np.linalg.norm(follow_vector)
            if distance > 0:
                follow_vector = follow_vector / distance
        else:
            follow_vector = np.array([ped.direction, 0])
        
        # 融合 ORCA、DWA 和跟随行为
        final_velocity = (
            0.4 * orca_velocity +  # ORCA 避碰
            0.3 * dwa_velocity +   # DWA 导航
            0.3 * follow_vector * ped.speed  # 跟随行为
        )
        
        # 确保速度不超过最大速度
        speed = np.linalg.norm(final_velocity)
        if speed > 3:
            final_velocity = final_velocity / speed * 3
        
        return final_velocity

    def _orca(self, ped, nearby_pedestrians):
        avoid_vector = np.zeros(2)
        for other in nearby_pedestrians:
            diff = np.array([ped.x - other.x, ped.y - other.y])
            dist = np.linalg.norm(diff)
            if dist > 1e-6:
                avoid_vector += diff / (dist ** 2) * 30

        avoid_magnitude = np.linalg.norm(avoid_vector)
        if avoid_magnitude > 1:
            avoid_vector = avoid_vector / avoid_magnitude

        new_velocity = np.array([ped.vx, ped.vy]) + avoid_vector
        
        speed = np.linalg.norm(new_velocity)
        if speed > 3:
            new_velocity = new_velocity / speed * 3
        
        return new_velocity
    

    def _dwa(self, ped, orca_velocity):
        goal_direction = np.array([1, 0]) if ped.direction == 1 else np.array([-1, 0])
        same_direction_pedestrians = [p for p in self.pedestrians if p.direction == ped.direction]
        if same_direction_pedestrians:
            center_y = np.mean([p.y for p in same_direction_pedestrians])
            y_diff = center_y - ped.y
        else:
            y_diff = 0
        
        # 添加墙壁避让逻辑
        wall_avoidance = 0
        if ped.x < self.sidewalk_width * 2:  # 接近左墙
            wall_avoidance = 1
        elif ped.x > self.width - self.sidewalk_width * 2:  # 接近右墙
            wall_avoidance = -1
        
        utility = (self.dwa_params['alpha'] * np.dot(orca_velocity, goal_direction) +
                   self.dwa_params['beta'] * y_diff * 0.01 -
                   self.dwa_params['gamma'] * self._distance_to_walls(ped) * 0.1 +
                   self.dwa_params['delta'] * self._distance_to_same_direction(ped) * 0.1 -
                   self.dwa_params['epsilon'] * self._distance_to_opposite_direction(ped) * 0.1 +
                   0.5 * wall_avoidance)  # 添加墙壁避让项
        
        utility = max(utility, 0.9)
        
        return orca_velocity * utility

    def _distance_to_walls(self, ped):
        return min(ped.y, self.height - ped.y)

    def _distance_to_same_direction(self, ped):
        return min([np.linalg.norm(np.array([ped.x - other.x, ped.y - other.y])) 
                    for other in self.pedestrians 
                    if other != ped and other.direction == ped.direction] or [float('inf')])

    def _distance_to_opposite_direction(self, ped):
        return min([np.linalg.norm(np.array([ped.x - other.x, ped.y - other.y])) 
                    for other in self.pedestrians 
                    if other != ped and other.direction != ped.direction] or [float('inf')])

    def _calculate_reward(self):
        following_reward = self._calculate_following_reward()
        flow_efficiency_reward = self._calculate_flow_efficiency()
        collision_penalty = self._calculate_collision_penalty()
        completion_reward = self.completed * 0.5
        lane_formation_reward = self._evaluate_lane_formation() * 2
        
        return (
            following_reward +
            flow_efficiency_reward +
            completion_reward +
            lane_formation_reward -
            collision_penalty -
            self.steps * 0.0005
        )
        
    def _evaluate_position_correctness(self):
        correct_positions = sum(1 for ped in self.pedestrians if 
                                (ped.direction == 1 and ped.y < self.height/2) or 
                                (ped.direction == -1 and ped.y >= self.height/2))
        return correct_positions / len(self.pedestrians)
    
    def _adjust_pedestrian_positions(self):
        for ped in self.pedestrians:
            if ped.direction == 1 and ped.y >= self.height/2:
                ped.y -= random.uniform(0, 1)  # Slowly move blue pedestrians upwards
            elif ped.direction == -1 and ped.y < self.height/2:
                ped.y += random.uniform(0, 1)  # Slowly move red pedestrians downwards
            
            # Ensure pedestrians stay within the environment
            ped.y = max(ped.radius, min(ped.y, self.height - ped.radius))


    def _calculate_flow_efficiency(self):
        left_flow = sum(1 for ped in self.pedestrians if ped.direction == -1 and ped.vx < 0)
        right_flow = sum(1 for ped in self.pedestrians if ped.direction == 1 and ped.vx > 0)
        
        # 奖励双向流动的平衡性
        flow_balance = 1 - abs(left_flow - right_flow) / len(self.pedestrians)
        
        # 计算平均速度
        avg_speed = np.mean([np.linalg.norm([ped.vx, ped.vy]) for ped in self.pedestrians])
        
        return flow_balance * 0.5 + avg_speed * 0.1
    
    def _calculate_following_reward(self):
        total_following = 0
        avg_following_distance = 0
        for ped in self.pedestrians:
            if ped.following:
                total_following += 1
                distance = np.linalg.norm(np.array([ped.x - ped.following.x, ped.y - ped.following.y]))
                avg_following_distance += distance

        if total_following > 0:
            avg_following_distance /= total_following
            # 奖励更多的跟随行为和更近的跟随距离
            return total_following * 0.1 - avg_following_distance * 0.01
        return 0 

    def _evaluate_lane_formation(self):
        top_blue = sum(1 for ped in self.pedestrians if ped.y < self.height/2 and ped.direction == 1)
        bottom_red = sum(1 for ped in self.pedestrians if ped.y >= self.height/2 and ped.direction == -1)
        top_red = sum(1 for ped in self.pedestrians if ped.y < self.height/2 and ped.direction == -1)
        bottom_blue = sum(1 for ped in self.pedestrians if ped.y >= self.height/2 and ped.direction == 1)
        
        # Calculate the strength of each possible lane formation
        blue_top_formation = (top_blue + bottom_red) / len(self.pedestrians)
        red_top_formation = (top_red + bottom_blue) / len(self.pedestrians)
        
        # Return the stronger of the two formations
        return max(blue_top_formation, red_top_formation)

    def _calculate_collision_penalty(self):
        collisions = 0
        for i, ped1 in enumerate(self.pedestrians):
            for ped2 in self.pedestrians[i+1:]:
                if np.linalg.norm(np.array([ped1.x - ped2.x, ped1.y - ped2.y])) < ped1.radius + ped2.radius:
                    collisions += 1
        return collisions * 0.5

    def _calculate_lane_violation_penalty(self):
        violations = sum(1 for ped in self.pedestrians if 
                         (ped.direction == 1 and ped.y >= self.height/2) or 
                         (ped.direction == -1 and ped.y < self.height/2))
        return violations * 0.1

    def _render_frame(self):
        if self.render_mode == "human":
            self.screen.fill((200, 200, 200))
            
            # 左右两侧的灰色区域
            pygame.draw.rect(self.screen, (150, 150, 150), (0, 0, self.sidewalk_width, self.height))
            pygame.draw.rect(self.screen, (150, 150, 150), (self.width - self.sidewalk_width, 0, self.sidewalk_width, self.height))

            for ped in self.pedestrians:
                color = (0, 0, 255) if ped.direction == 1 else (255, 0, 0)
                x = max(0, min(int(ped.x), self.width))
                y = max(0, min(int(ped.y), self.height))
                pygame.draw.circle(self.screen, color, (x, y), ped.radius)
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            pass

    def _find_leader(self, ped):
        same_direction_peds = [p for p in self.pedestrians if p.direction == ped.direction and p != ped]
        if not same_direction_peds:
            return None
        
        # 找到所有前方的智能体
        if ped.direction == 1:
            potential_leaders = [p for p in same_direction_peds if p.x > ped.x]
        else:
            potential_leaders = [p for p in same_direction_peds if p.x < ped.x]
        
        if not potential_leaders:
            return None
        
        # 找到最近的前方智能体，考虑x和y方向的距离
        nearest_leader = min(potential_leaders, key=lambda p: (p.x - ped.x)**2 + (p.y - ped.y)**2)
        
        # 检查总距离，如果太远就不跟随
        distance = np.sqrt((nearest_leader.x - ped.x)**2 + (nearest_leader.y - ped.y)**2)
        if distance > 100:  # 可以根据需要调整这个阈值
            return None
        
        return nearest_leader

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

class Pedestrian:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = random.uniform(1, 2)
        self.vx = self.speed * direction
        self.vy = 0
        self.radius = 5
        self.total_distance = 0
        self.time_elapsed = 0
        self.following = None
        self.follow_distance = 30

    def move(self):
        self.x += self.vx
        self.y += self.vy
        self.x = max(self.radius, min(self.x, 800 - self.radius))
        self.y = max(self.radius, min(self.y, 400 - self.radius))

    def adjust_speed_to_follow(self, leader):
        if leader:
            dx = leader.x - self.x
            dy = leader.y - self.y
            distance = np.sqrt(dx**2 + dy**2)
            
            # 计算目标方向
            target_direction = np.array([dx, dy]) / distance if distance > 0 else np.array([0, 0])
            
            # 调整速度
            if distance > self.follow_distance:
                self.speed = min(self.speed * 1.1, 2)  # 加速，但不超过最大速度
            elif distance < self.follow_distance * 0.8:
                self.speed = max(self.speed * 0.9, 1)  # 减速，但不低于最小速度
            
            # 更新速度向量
            self.vx = self.speed * target_direction[0]
            self.vy = self.speed * target_direction[1]
        else:
            # 如果没有领导者，保持原来的方向
            self.vx = self.speed * self.direction
            self.vy = 0

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.01
        self.device = torch.device("cuda")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episodes, batch_size=32, steps_per_episode=100000):
    logging.basicConfig(filename='pedestrian_flow.log', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    env = PedestrianFlowEnv(render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    print(f"Device is {agent.device}")

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for time in range(steps_per_episode):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.train(batch_size)

            if done:
                break

        logging.info(f"Episode: {e+1}, Score: {total_reward}, Epsilon: {agent.epsilon:.4f}, Total Steps: {env.total_steps}")

        if (e + 1) % 10 == 0:
            print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.4f}, Total Steps: {env.total_steps}")

        if (e + 1) % 100 == 0:
            torch.save(agent.model.state_dict(), f'dqn_model_episode_{e+1}.pth')

    env.close()

if __name__ == "__main__":
    train_dqn(episodes=1000)