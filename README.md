# Deep Q-Learning for Atari Pong

This repository contains an implementation of a Deep Q-Network (DQN) agent that learns to play the Atari game Pong by extracting features directly from RAM memory rather than processing raw pixel data.

## Demo Video

![Pong Agent Demo](data/pong.mp4)

## Project Overview

Traditional approaches to Atari game playing with reinforcement learning typically use image data as input, which requires significant computational resources. This project demonstrates that feature extraction from RAM can be a computationally efficient alternative while achieving good performance.

The agent uses a standard DQN architecture with experience replay and target networks to learn a successful Pong-playing policy through trial and error.

## Features

- **RAM-based Feature Extraction**: Extracts key game state information directly from RAM (paddle positions, ball coordinates) rather than processing pixels
- **Frame Stacking**: Uses a history of 4 consecutive frames to capture motion information
- **Experience Replay**: Stores and reuses past experiences to improve sample efficiency
- **Target Network**: Uses a separate target network for stable learning
- **Epsilon-greedy Exploration**: Balanced exploration-exploitation strategy
- **Comprehensive Metrics Tracking**: Monitors rewards, loss, episode duration, and more throughout training

## Usage

### Training

To train the agent from scratch:

```bash
python pong_train.py
```

Training will automatically save the model and metrics every 10 episodes and can be resumed if interrupted.

### Testing

To test the trained agent:

```bash
python pong_test.py
```

## Implementation Details

### State Representation
The agent uses a compact state representation consisting of:
- Right paddle position (RAM index 51)
- Left paddle position (RAM index 50)
- Ball x-coordinate (RAM index 49)
- Ball y-coordinate (RAM index 54)

These values are normalized and stacked across 4 frames, resulting in a 16-dimensional state vector.

### Network Architecture
- Simple feedforward neural network with two hidden layers (128 and 64 neurons)
- ReLU activations
- Output layer producing Q-values for each possible action

### Hyperparameters
- Learning rate: 0.001
- Discount factor (gamma): 0.99
- Initial exploration rate (epsilon): 1.0
- Epsilon decay: 0.9975
- Minimum epsilon: 0.01
- Replay buffer size: 100,000
- Batch size: 32
- Target network update frequency: 1000 steps

## Results

The agent demonstrates clear learning progress over training:
- Initial rewards around -20 (losing most points)
- Improved to around -15 to -10 by later training
- Occasional positive rewards reaching as high as +5 (winning rounds)
- Episode durations increasing from ~1000 to ~4000-6000+ steps
- Reward per step improving from -0.025 to nearly 0.0 in later episodes
- Training reaching over 2000 episodes and accumulating 6+ million steps
- The model starts overfitting after 2000 episodes, with rewards stabilizing around -10 to -5

## Future Improvements

Potential areas for enhancement:
- Adding more derived features like ball direction vectors
- Implementing prioritized experience replay
- Adding learning rate scheduling
- Exploring advanced techniques like Double DQN or Dueling DQN
- Optimizing target network update frequency

## Acknowledgments

- OpenAI Gymnasium for the Atari Pong environment
- PyTorch team for the deep learning framework
- Mnih et al. for the original DQN algorithm