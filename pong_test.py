import os
import torch
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from pong_train import DQN, PongEnv
import random

SAVE_DIR = "data/"
MODEL_PATH = os.path.join(SAVE_DIR, "dqn_model.pth")

class TestAgent:
    def __init__(self):
        """Initialize the environment and load the trained model."""
        self.env = PongEnv(render_mode="human")
        self.state_dim = 4  # Right paddle, left paddle, ball X, ball Y
        self.action_dim = self.env.env.action_space.n
        self.epsilon = 0.1

        # Load trained model
        self.model = DQN(self.state_dim, self.action_dim)
        self.load_model()

    def load_model(self):
        """Load the trained model from file."""
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.model.eval()  # Set model to evaluation mode
            print(f"Loaded trained model from {MODEL_PATH}")
        else:
            raise FileNotFoundError("No trained model found! Train and save a model first.")

    def play(self, num_games=1):
        """Play a test game using the trained model."""
        for game in range(num_games):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated = self.env.step(action)
                total_reward += reward
                state = next_state
                done = terminated or truncated  # End of episode
            
            print(f"Game {game + 1}: Total Reward = {total_reward:.2f}")

        self.env.close()
    
    def select_action(self, state):
        """Epsilon-greedy policy for action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.model.network[-1].out_features - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.model(state_tensor)).item()


if __name__ == "__main__":
    tester = TestAgent()
    num_games = 5
    tester.play(num_games)
