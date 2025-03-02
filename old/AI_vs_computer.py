import gym
from dqnatari import DQNAtari

# Construct the Atari environment and load the Breakout game.
env = gym.make('Pong-v4', render_mode='human')

# Initialize the DQN model with the Breakout environment.
# Non-essential portions of the screen are cropped out.
dqn = DQNAtari("Pong", env, image_crop=(0, 34, 160, 194))

dqn.load_model(steps=1000000)

# Play the game in real time.
dqn.play(
    episodes=1,                 # The number of episodes to play.
    eps=0.0,                    # The eps to be used during gameplay.
    fps=30,                     # The speed od the game.
    save_frames=False           # Save each individual frame to a .png file.
)
