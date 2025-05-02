import os
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
import pickle
import signal
import sys

SAVE_DIR = "data/"
LOG_DIR = "logs/"
MODEL_PATH = os.path.join(SAVE_DIR, "dqn_model.pth")
METRICS_FILE = os.path.join(LOG_DIR, "training_metrics.npz")
PLOT_FILE = os.path.join(LOG_DIR, "training_metrics_plot.png")
WINDOW_SIZE = 4
SAVE_FREQUENCY = 10

class PongEnv:
    def __init__(self, render_mode="human"):
        self.env = gym.make("ALE/Pong-v5", render_mode=render_mode, obs_type="ram")
        self.frame_stack = deque(maxlen=WINDOW_SIZE)
        self.key_features = {
            "paddle_right": 51,  # Right paddle position
            "paddle_left": 50,   # Left paddle position
            "ball_x": 49,        # Ball X
            "ball_y": 54         # Ball Y
        }

    def extract_features(self, ram):
        return np.array([ram[idx] / 255.0 for idx in self.key_features.values()], dtype=np.float32)

    def reset(self):
        ram, _ = self.env.reset()
        initial_frame = self.extract_features(ram)
        self.frame_stack.clear()
        for _ in range(WINDOW_SIZE):
            self.frame_stack.append(initial_frame)
        return np.array(self.frame_stack).flatten()

    def step(self, action):
        ram, reward, terminated, truncated, _ = self.env.step(action)
        self.frame_stack.append(self.extract_features(ram))
        return np.array(self.frame_stack).flatten(), reward, terminated, truncated

    def close(self):
        self.env.close()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
        super(DQN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class RLAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.995, min_epsilon=0.01, memory_size=100000, batch_size=32,
                 target_update_freq=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory = deque(maxlen=memory_size)
        self.update_counter = 0
        self.current_episode = 0
        self.memory_size = memory_size
        
        self.policy_model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.update_target_network()
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.policy_model.network[-1].out_features - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.policy_model(state_tensor)).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        q_values = self.policy_model(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()
            
        return loss.item()

    def save_buffer(self):
        if len(self.memory) > 0:
            buffer_path = os.path.join(SAVE_DIR, "replay_buffer.pkl")
            try:
                with open(buffer_path, 'wb') as f:
                    pickle.dump(list(self.memory), f)
                print(f"Saved all {len(self.memory)} experiences to replay buffer")
            except Exception as e:
                print(f"Error saving replay buffer: {e}")
    
    def load_buffer(self):
        buffer_path = os.path.join(SAVE_DIR, "replay_buffer.pkl")
        if os.path.exists(buffer_path):
            try:
                with open(buffer_path, 'rb') as f:
                    loaded_buffer = pickle.load(f)
                
                self.memory.clear()
                for experience in loaded_buffer:
                    self.memory.append(experience)
                    
                print(f"Loaded {len(self.memory)} experiences to replay buffer")
                return True
            except Exception as e:
                print(f"Error loading replay buffer: {e}")
        return False
    
    def save_model(self):
        """Save model and buffer"""
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save({
            "policy_model_state_dict": self.policy_model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "current_episode": self.current_episode,
            "update_counter": self.update_counter
        }, MODEL_PATH)
        
        self.save_buffer()
    
    def load_model(self):
        """Load model and buffer"""
        model_loaded = False
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
            self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint["epsilon"]
            self.current_episode = checkpoint["current_episode"]
            self.update_counter = checkpoint.get("update_counter", 0)
            model_loaded = True
        
        self.load_buffer()
        
        return model_loaded

class MetricsLogger:
    def __init__(self, current_episode=0):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.metrics = {
            "episodes": [],
            "rewards": [],
            "epsilons": [],
            "losses": [],
            "episode_durations": [],
            "rewards_per_step": [],
            "total_steps": []
        }
        self.total_steps = 0
        self._load_metrics(current_episode)
    
    def _load_metrics(self, current_episode):
        if not os.path.exists(METRICS_FILE):
            return
            
        try:
            data = np.load(METRICS_FILE, allow_pickle=True)
            all_episodes = data['episode'].tolist()
            
            if len(all_episodes) > 0:
                last_checkpoint_episode = ((current_episode - 1) // SAVE_FREQUENCY) * SAVE_FREQUENCY
                valid_idx = [i for i, ep in enumerate(all_episodes) if ep <= last_checkpoint_episode]
                
                if valid_idx:
                    last_idx = max(valid_idx)
                    self.metrics["episodes"] = all_episodes[:last_idx+1]
                    self.metrics["rewards"] = data['reward'][:last_idx+1].tolist()
                    self.metrics["epsilons"] = data['epsilon'][:last_idx+1].tolist()
                    self.metrics["episode_durations"] = data['duration'][:last_idx+1].tolist()
                    self.metrics["rewards_per_step"] = data['reward_per_step'][:last_idx+1].tolist()
                    
                    if 'total_steps' in data:
                        self.metrics["total_steps"] = data['total_steps'][:last_idx+1].tolist()
                        if self.metrics["total_steps"]:
                            self.total_steps = self.metrics["total_steps"][-1]
                    else:
                        total = 0
                        self.metrics["total_steps"] = []
                        for steps in self.metrics["episode_durations"]:
                            total += steps
                            self.metrics["total_steps"].append(total)
                        self.total_steps = total
                    
                    if 'avg_loss' in data:
                        self.metrics["losses"] = data['avg_loss'][:last_idx+1].tolist()
        except Exception as e:
            print(f"Error loading metrics: {e}")
    
    def log_episode(self, episode, reward, epsilon, avg_loss, duration, total_steps=None):
        self.metrics["episodes"].append(episode)
        self.metrics["rewards"].append(reward)
        self.metrics["epsilons"].append(epsilon)
        if avg_loss is not None:
            self.metrics["losses"].append(avg_loss)
        self.metrics["episode_durations"].append(duration)
        self.metrics["rewards_per_step"].append(reward / max(duration, 1))
        
        if total_steps is not None:
            self.total_steps = total_steps
        else:
            self.total_steps += duration
        
        self.metrics["total_steps"].append(self.total_steps)
        
        if (episode + 1) % SAVE_FREQUENCY == 0:
            self.save()
    
    def save(self):
        data = {
            "episode": np.array(self.metrics["episodes"]),
            "reward": np.array(self.metrics["rewards"]),
            "epsilon": np.array(self.metrics["epsilons"]),
            "duration": np.array(self.metrics["episode_durations"]),
            "reward_per_step": np.array(self.metrics["rewards_per_step"]),
            "total_steps": np.array(self.metrics["total_steps"])
        }
        if self.metrics["losses"]:
            data["avg_loss"] = np.array(self.metrics["losses"])
        
        np.savez(METRICS_FILE, **data)
        self._plot()
    
    def _plot(self):
        plt.figure(figsize=(15, 15))
        plot_configs = [
            ("rewards", "Episode Rewards", "Reward"),
            ("epsilons", "Exploration Rate (Epsilon)", "Epsilon"),
            ("losses", "Training Loss", "Loss"),
            ("episode_durations", "Episode Duration", "Steps"),
            ("rewards_per_step", "Reward Per Step", "Reward/Step"),
            ("total_steps", "Cumulative Steps", "Total Steps")
        ]
        
        base_len = len(self.metrics["episodes"])
        
        for i, (metric_key, title, ylabel) in enumerate(plot_configs):
            if metric_key == "losses" and not self.metrics["losses"]:
                continue
                
            if len(self.metrics[metric_key]) != base_len:
                print(f"Warning: {metric_key} array length ({len(self.metrics[metric_key])}) " 
                      f"doesn't match episodes array length ({base_len}). Skipping plot.")
                continue
                
            plt.subplot(3, 2, i+1)
            plt.plot(self.metrics["episodes"], self.metrics[metric_key])
            plt.title(title)
            plt.xlabel("Episode")
            plt.ylabel(ylabel)
        
        plt.tight_layout()
        plt.savefig(PLOT_FILE)
        plt.close()

def clip_reward(reward, clip_value=1.0):
    return max(min(reward, clip_value), -clip_value)

def train(num_episodes=10000, render_mode="rgb_array", clip_rewards=True):
    def signal_handler(sig, frame):
        print("\nTraining interrupted. Saving progress...")
        agent.save_model()
        metrics_logger.save()
        pong_env.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    pong_env = PongEnv(render_mode=render_mode)
    state_dim = 4 * WINDOW_SIZE
    action_dim = pong_env.env.action_space.n
    agent = RLAgent(state_dim, action_dim, target_update_freq=1000)

    agent.load_model()
    current_episode = agent.current_episode
    initial_epsilon = agent.epsilon

    if len(agent.memory) >= agent.batch_size:
        print(f"Pre-training with loaded experiences ({len(agent.memory)} samples available)...")
        for _ in range(10):
            agent.train()
    
    metrics_logger = MetricsLogger(current_episode=current_episode)
    
    if metrics_logger.metrics["episodes"] and metrics_logger.metrics["episodes"][-1] < current_episode - 1:
        metrics_logger.metrics["episodes"].append(current_episode - 1)
        metrics_logger.metrics["epsilons"].append(initial_epsilon)
        metrics_logger.metrics["rewards"].append(0)
        metrics_logger.metrics["episode_durations"].append(0)
        metrics_logger.metrics["rewards_per_step"].append(0)
        metrics_logger.metrics["total_steps"].append(metrics_logger.total_steps)
        
        if metrics_logger.metrics["losses"]:
            metrics_logger.metrics["losses"].append(0)

    total_steps = metrics_logger.total_steps
    
    for episode in range(current_episode, num_episodes):
        agent.current_episode = episode
        state = pong_env.reset()
        episode_start_time = time.time()
        episode_reward = 0
        episode_steps = 0
        episode_losses = []

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated = pong_env.step(action)
            
            episode_reward += reward
            
            if clip_rewards:
                clipped_reward = clip_reward(reward)
            else:
                clipped_reward = reward
                
            agent.store_experience(state, action, clipped_reward, next_state, terminated)
            state = next_state
            
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
                
            episode_steps += 1

            if terminated or truncated:
                agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
                break
        
        total_steps += episode_steps
        episode_duration = time.time() - episode_start_time
        avg_loss = np.mean(episode_losses) if episode_losses else None
        
        metrics_logger.log_episode(
            episode, episode_reward, agent.epsilon, avg_loss, episode_steps, total_steps
        )
        
        print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.1f} | "
              f"Steps = {episode_steps} | Total = {total_steps} | "
              f"Time = {episode_duration:.1f}s | Epsilon = {agent.epsilon:.3f}")

        if (episode + 1) % SAVE_FREQUENCY == 0:
            agent.save_model()

    agent.save_model()
    pong_env.close()

if __name__ == "__main__":
    train()