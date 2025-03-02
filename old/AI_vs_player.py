import pygame
from pettingzoo.atari.pong import pong
from dqnatari import DQNAtari
import random

# Construct the Atari environment and load the Breakout game.
#env = gym.make('Pong-v4', render_mode='human')
env = pong.env()

# Initialize the DQN model with the Breakout environment.
# Non-essential portions of the screen are cropped out.
dqn = DQNAtari("Pong", env, image_crop=(0, 34, 160, 194))
dqn.load_model(steps=1000000)

env.reset()
clock = pygame.time.Clock()
window = []
eps = 0.01
user_action = 0
done = False

while not done:
    env.render(mode='human')
    observation, reward, done, info = env.last()

    if done:
        continue

    frame = dqn.process_observation(observation)

    if len(window) < DQNAtari.WINDOW_SIZE:
        window.append(frame)
    else:
        for i in range(DQNAtari.WINDOW_SIZE - 1):
            window[i] = window[i+1]
        window[DQNAtari.WINDOW_SIZE - 1] = frame

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                user_action = 1
            if event.key == pygame.K_UP:
                user_action = 2
            if event.key == pygame.K_DOWN:
                user_action = 3
            if event.key == pygame.K_ESCAPE:
                quit()
        elif event.type == pygame.KEYUP:
            user_action = 0
        
    if env.agent_selection == "second_0":
        action = user_action
    else:
        if random.random() < eps or len(window) < DQNAtari.WINDOW_SIZE:
            action = random.randint(0, 3)
        else:
            action = dqn.get_policy_action(window)
            #action = random.randint(0, 3)

    env.step(action)
    clock.tick(120)

