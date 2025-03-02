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

SAVE_DIR = "data/"
MODEL_PATH = os.path.join(SAVE_DIR, "dqn_model.pth")

class PongEnv:
    def __init__(self, render_mode="human"):
        """Initialize the Pong environment using RAM observations."""
        self.env = gym.make("ALE/Pong-v5", render_mode=render_mode, obs_type="ram")

    def extract_features(self, ram):
        """Extract right paddle, left paddle, ball X, and ball Y positions from RAM and normalize."""
        right_paddle = ram[51] / 255.0
        left_paddle = ram[50] / 255.0
        ball_x = ram[49] / 255.0
        ball_y = ram[54] / 255.0
        return np.array([right_paddle, left_paddle, ball_x, ball_y], dtype=np.float32)

    def reset(self):
        """Reset the environment and return the extracted features."""
        ram, _ = self.env.reset()
        return self.extract_features(ram)

    def step(self, action):
        """Take a step and return extracted features."""
        ram, reward, terminated, truncated, _ = self.env.step(action)
        return self.extract_features(ram), reward, terminated, truncated

    def close(self):
        """Close the environment."""
        self.env.close()


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class RLAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.995, min_epsilon=0.01, memory_size=100000, batch_size=32): # domen max_size=1000000
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.current_episode = 0

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience for replay."""
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, state):
        """Epsilon-greedy policy for action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.model.network[-1].out_features - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.model(state_tensor)).item()

    def train(self):
        """Train the agent using experience replay."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        q_values = self.model(states).gather(1, actions).squeeze()
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_model(self):
        """ Save the model and optimizer state. """
        os.makedirs(SAVE_DIR, exist_ok=True)  # Ensure save directory exists

        # Save model weights and optimizer state
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "current_episode": episode + 1
        }, MODEL_PATH)

        print(f"Model and optimizer state saved to {MODEL_PATH}")



    def load_model(self):
        """Load the model safely, restoring necessary training components."""
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
            
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint["epsilon"]
            self.current_episode = checkpoint["current_episode"]

            print(f"Model and optimizer state loaded from {MODEL_PATH}")
        else:
            print("No saved model found, starting fresh.")




if __name__ == "__main__":
    # Choose render_mode = "human" for visualization, "rgb_array" for headless training
    pong_env = PongEnv(render_mode="rgb_array")
    state_dim = 4
    action_dim = pong_env.env.action_space.n
    agent = RLAgent(state_dim, action_dim)

    # Load previous progress if available
    agent.load_model()

    num_episodes = 10000

    
    for episode in range(agent.current_episode, num_episodes):
        state = pong_env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated = pong_env.step(action)
            agent.store_experience(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            agent.train()

            if terminated or truncated:
                agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
                break
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}")

        # Save after each episode
        agent.save_model()

    pong_env.close()