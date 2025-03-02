import gym
from dqnatari import DQNAtari
import random

#env = gym.make('Pong-v4', render_mode='human')
env = gym.make('Pong-v4')

# Initialize the DQN model with the Breakout environment.
# Non-essential portions of the screen are cropped out.
dqn = DQNAtari("Pong", env, num_actions=4, image_crop=(0, 34, 160, 194))

# Train the DQN.
dqn.train(
    training_steps=1000000,     # The number of training steps.
    memory_size=1000000,        # The number of experiences that the replay memory can store.
    warm_up_steps=50000,        # The number of steps before the training stars.
    target_update_steps=10000,  # How often should the target network be updated.
    model_save_steps=100000,    # How often should the policy network be saved to disk.
    epoch_steps=10000,          # The length of one epoch.
    action_duration=1,          # The duration of each action.
    eps_max=1.0,                # The starting eps.
    eps_min=0.1,                # The final eps.
    eps_interval=1000000,       # When the final eps is reached.
    clip_reward=True,           # Clip the rewards to {-1, 0, 1}.
    log_episodes=True           # Save the information of each episode to a file.
)
