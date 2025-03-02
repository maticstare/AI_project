import os, time, random
import numpy as np
from PIL import Image

# Do not print Tensorflow debug messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Permute, Convolution2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam

class ReplayMemory:
    """
    Stores the experiences obtained during training. When the number of experiences exceeds the set
    capacity (typically 1M experiences), the oldest experiences are overwritten with new ones.

    The conceptual structure of the replay memory:
    ---------------------------------
    |          experience 0         |
    ---------------------------------
    |          experience 1         |
    ---------------------------------
    |              ...              |
    ---------------------------------
    |          experience n         |
    ---------------------------------

    experience i:
    ---------------------------------
    |            window i           |
    ---------------------------------
    |         action i (int)        |
    ---------------------------------
    |        reward i (float)       |
    ---------------------------------
    |          window i + 1         |
    ---------------------------------
    |  terminal state i + 1 (bool)  |
    ---------------------------------

    window:
    ---------------------------------
    |          frame k + 0          |
    ---------------------------------
    |          frame k + 1          |
    ---------------------------------
    |          frame k + 2          |
    ---------------------------------
    |          frame k + 3          |
    ---------------------------------

    frame:
    ------------------------------------
    numpy array 84 x 84 x uint8 [0, 255]
    ------------------------------------

    To preserve memory, the actual implementation is like this:
    ---------------------------------
    |            entry 0            |
    ---------------------------------
    |            entry 1            |
    ---------------------------------
    |              ...              |
    ---------------------------------
    |            entry n            |
    ---------------------------------

    entry 0:
    ---------------------------------
    |               0               |
    ---------------------------------
    |               0               |
    ---------------------------------
    |            frame 0            |
    ---------------------------------
    |             False             |
    ---------------------------------

    entry i > 0:
    ---------------------------------
    |         action i (int)        |
    ---------------------------------
    |        reward i (float)       |
    ---------------------------------
    |          frame i + 1          |
    ---------------------------------
    |  terminal state i + 1 (bool)  |
    ---------------------------------

    frame:
    ------------------------------------
    numpy array 84 x 84 x uint8 [0, 255]
    ------------------------------------

    When retrieving the i-th experience from the replay memory, the experience is constructed from k
    consequent entries, according to the window size.
    """

    def __init__(self, capacity, window_size, initial_frame):
        """
        Parameters:
            capacity:      The capacity of the replay memory (the number of experiences).
            window_size:   The number of frames in a window.
            initial_frame: The initial frame (after the environment has been reset).
        """
        self._window_size = window_size
        self._entries_capacity = capacity + window_size
        self._entries = [(0, 0, initial_frame, False)]
        self._entry_index = 1 # The index of the next entry.

    def add_experience(self, action, reward, frame, terminal):
        """
        Adds a new experience to the replay memory. Only the latest frame is given, the rest can
        be found in older entries.

        The i-th window is composed of frames (i - window_size + 1), ..., i.
        The entries buffer is circular, so a window can span over the edges of the full buffer.
        Indices to the entries buffer are therefore always taken as modulo with the capacity.
        
        Parameters:
            action (int):        The index of the action taken.
            reward (float):      The reward obtained by taking the given action.
            frame (numpy.array): The new state - a 84 x 84 x uint8 numpy array.
            terminal (boolean):  True if the given state (frame) is a terminal state (game over).
        """
        if (len(self._entries) < self._entries_capacity):
            self._entries.append((action, reward, frame, terminal))
        else:
            self._entries[self._entry_index] = (action, reward, frame, terminal)
        self._entry_index = (self._entry_index + 1) % self._entries_capacity

    def experience_count(self):
        """
        Returns the number of experiences in the replay memory. Since we store entries and not
        experiences, the number of experiences must be computed according to the window size.

        Returns:
            count (int): The number of stored experiences.
        """
        count = len(self._entries) - self._window_size
        return count if count >= 0 else 0

    def get_last_experience(self):
        """
        Constructs and returns the latest experience from the replay buffer.

        Returns:
            experience: The last experience from the replay buffer.
        """
        assert self.experience_count() > 0
        entry_index = (self._entry_index - self._window_size - 1) % self._entries_capacity
        return self._construct_experience(entry_index)

    def get_random_experiences(self, batch_size=32):
        """
        Constructs and returns a random batch (a python array) of experiences.
        
        Parameters:
            batch_size (int): The size of the batch.
        Returns:
            batch (array): A batch of random experiences.
        """
        batch = []
        for experience_index in random.sample(range(0, self.experience_count()), batch_size):
            entry_index = (experience_index + self._entry_index - len(self._entries)) % self._entries_capacity
            batch.append(self._construct_experience(entry_index))
        return batch

    def _construct_experience(self, entry_index):
        """
        Constructs and returns the i-th experience. There must be a sufficient number of entries
        in the memory buffer (at least window_size + 1).

        The i-th experience contains two windows.
        window0: window i
        window1: window (i + 1)

        Given an entry index k, window0 is reconstructed from frames [k,...,(k + window_size)],
        and window1 from frames [(k + 1),...,(k + 1 + window_size)].

        Action and reward of the experience are taken from the last entry (k + 1 + window_size).

        A window should not span over two distinct episodes. The end of an episode is determined
        by the terminal flag. If a terminal frame is encountered in the middle of constructing a
        window, this terminal frame is extended to the rest of the window.

        Parameters:
            entry_index (int): The index of the entry at which to start constructing the experience.
        Returns:
            experience: The constructed experience.
        """
        window0 = []
        window1 = []

        use_entries = True # Stop using new entries if a terminal frame is encountered.
        for i in range(0, self._window_size + 1):    
            if use_entries:               
                # The _entry_index is the point at which new entries overwrite old entries.
                # A window should never span over that point.
                buffer_index = (entry_index + i) % self._entries_capacity
                assert i == 0 or buffer_index != self._entry_index
                
                (action, reward, state, terminal) = self._entries[buffer_index]

                if terminal:
                    use_entries = False
            
            if i < self._window_size:
                window0.append(state)
            if i > 0:
                window1.append(state)

        return (window0, action, reward, window1, terminal)

class GameTimer:
    """
    Provides a simple timer to track the deadline for rendering the next frame.
    """
    def __init__(self, fps):
        """
        Parameters:
            fps (int): The game frame rate, used to compute the timing for a single frame.
        """
        self._frame_duration_ms = 1000 / fps
        self._next_frame_timestamp = time.time_ns() + self._frame_duration_ms * 1000000
    
    def wait_next_frame(self):
        """
        Waits until the deadline to render the next gameplay frame.
        """
        while time.time_ns() < self._next_frame_timestamp:
            time.sleep(0.001)
        self._next_frame_timestamp += self._frame_duration_ms * 1000000

class DQNAtari:
    """
    Implementation of DQN as used by Atari games.

            window           ------------------
    (frame 0,...,frame n) -> | Neural Network | -> Q-values (action 0,...,action m)
                             ------------------

    Each frame is a numpy array 84 x 84 of floats [0.0, 1.0]. This differs from the replay memory
    frames which store bytes (to preserve the memory). before feeding a window to a network, it
    needs to be converted from uint8 to float and each pixel divided by 255.

    The output is an array of Q-values for each action. The action with the highest Q-value is
    executed in the given state (window).

    There are two neural networks used:
    - The policy network to predict the best action to take next.
    - The target network to help predict the target Q-values during training.

    While training, the target network is updated periodically: target network <- policy network.
    After the training is finished, only the policy network is used to play the game.
    """
    WINDOW_SIZE = 4
    INPUT_SHAPE = (WINDOW_SIZE, 84, 84)

    def __init__(self, name, gym_env, num_actions=4, image_crop=None):
        """
        Constructs and initializes the policy and target networks.

        Parameters:
            name:        An arbitrary name for this DQN. Used to name the files to store various data.
            gym_env:     Gym-compatible environment that contains at least the following methods:
                            - reset() -> observation (numpy array X x Y x [uint8, uint8, uint8]).
                            - step(action) -> observation, reward, done, info.
            num_actions: Number of actions. Actions are determined by indices 0, ..., num_actions - 1.
            image_crop:  A rectangular region to crop input images. If None, the whole image is used.
                         The value is in the form (x0, y0, x1, y1).
        """
        self._name = name
        self._gym_env = gym_env
        self._num_actions = num_actions
        self._image_crop = image_crop

        self._policy_network = Sequential()
        self._policy_network.add(Permute((2, 3, 1), input_shape=DQNAtari.INPUT_SHAPE))
        self._policy_network.add(Convolution2D(32, (8, 8), strides=(4, 4)))
        self._policy_network.add(Activation('relu'))
        self._policy_network.add(Convolution2D(64, (4, 4), strides=(2, 2)))
        self._policy_network.add(Activation('relu'))
        self._policy_network.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        self._policy_network.add(Activation('relu'))
        self._policy_network.add(Flatten())
        self._policy_network.add(Dense(512))
        self._policy_network.add(Activation('relu'))
        self._policy_network.add(Dense(num_actions))
        self._policy_network.add(Activation('linear'))
        self._policy_network.compile(optimizer=Adam(learning_rate=0.00025), loss="mse", metrics=['mae'])
    
        self._target_network = tf.keras.models.clone_model(self._policy_network)
        self._update_target_network()

        self._replay_memory = None

    def process_observation(self, observation, save_file=None):
        """
        Converts an image as obtained from the Gym environment to a frame. This includes cropping and
        resizing the image, and finally converting it to an 84 x 84 uint8 type numpy array as stored in
        the replay memory.

        Parameters:
            observation: The input image.
            save_file:   The name of the file to save the input image to.
                         If None, the image is not saved.
        Returns:
            frame: The frame to be stored in the replay buffer.
        """
        img = Image.fromarray(observation)
        if save_file != None:
            img.save(save_file)
        
        if self._image_crop != None:
            img = img.crop(self._image_crop)
        img = img.resize((84, 84)).convert('L')

        return np.array(img).astype('uint8')

    def load_model(self, steps=None):
        """
        Loads the weights of the neural network from a file. The names of saved files are composed
        of the given DQN name followed by the number of training at which the model was saved. When
        loading a model, we may provide the number of training steps. If the number of steps is not
        provided, it is omitted from the name of the file.

        The policy and the training network are both loaded from the same file.

        Parameters:
            steps (int): The number of training steps at which the model was saved.
        """
        if steps != None:
            filename = self._name + str(steps) + ".h5f"
        else:
            filename = self._name + ".h5f"
        self._policy_network = tf.keras.models.load_model(filename)
        self._update_target_network()

    def save_model(self, steps=None):
        """
        Saves the weights of the policy network in a file. The name of the file is composed of the
        given DQN name followed by the number of training steps at which the model is being saved.
        If the number of steps is not provided, it is omitted from the name of the file.

        Parameters:
            steps (int): The number of training steps at which the model is being saved.
        """
        if steps != None:
            filename = self._name + str(steps) + ".h5f"
        else:
            filename = self._name + ".h5f"
        self._policy_network.save(filename)

    def print_model(self):
        """
        Prints out the summary of the policy network.
        """
        print(self._policy_network.summary())
    
    def get_policy_action(self, window=[]):
        """
        Returns the best action for execution according to the current policy network. If a window is
        given, the action is returned for that window. Otherwise the most recent window is retrieved
        from the replay memory.

        A training method will typically store the frames is the replay memory and thus provide no
        window to this method. A testing/playing method would typically provide its current window.

        Parameters:
            window: The window for which to return the action or [] for the most recent window.
        Returns:
            action (int): Action to be executed for the given window.
        """
        if window == []:
            experience = self._replay_memory.get_last_experience()
            if experience == None:
                return None
            (_, _, _, window, _) = experience

        q_actions = self._policy_network.predict_on_batch(np.array([window]).astype('float')/255.0).flatten()
        action = np.argmax(q_actions)
        return action

    def train(self,
        training_steps=1000000,
        memory_size=1000000,
        warm_up_steps=50000,
        target_update_steps=10000,
        model_save_steps=100000,
        epoch_steps=10000,
        action_duration=1,
        eps_max=1.0,
        eps_min=0.1,
        eps_interval=1000000,
        clip_reward=True,
        log_episodes=True
    ):
        """
        Trains the model with the environment given with the constructor. The method initializes the replay
        memory, which is destroyed when training is finished.

        The training algorithm is a follows:

        for each training step do:
            1. Choose an action according to the current policy network with the eps
               probability of a random choice.
            2. Execute the chosen action and observe the next frame, the obtained reward,
               and whether the next state is terminal.
            3. Store the obtained frame, reward, and terminal flag to the replay memory.
            4. Periodically log the statistical data.
            5. If warming up, continue with the next step.
            6. Train the model on a random batch of experiences from the replay memory.
               (See the _train_on_batch method.)
            7. Periodically update the target network (target network <- policy network).
            8. Periodically save the policy network.

        Parameters:
            training_steps (int):      The number of steps to train the model.
            memory_size(int):          The number of experiences that the replay memory can store.
            warm_up_steps (int):       The number of steps at the start of training when only the experiences are
                                       stored in the replay memory, but the actual training is not yet in progress.
            target_update_steps (int): The intervals at which the target network is updated.
            model_save_steps (int):    The intervals at which the policy network is saved to a file.
            epoch_steps (int):         The length of one epoch. An epoch is used only for statistical reasons.
                                       At the end of each epoch the average epoch award and eps value is reported.
            action_duration (int):     The duration of actions in steps. If greater than 1, the same action will
                                       be executed over multiple game steps. One training step spans over the duration
                                       of one action, so this parameter also affects the duration of training.
            eps_max (float):           The eps value at the start of training.
            eps_min (float):           The eps value when the end of the [0, eps_interval] interval is reached.
            eps_interval (int):        The number of steps at which the eps interval ends. Up to that point the eps
                                       value decreases linearly. After this interval it stays at eps_min.
            clip_reward (bool):        If True, the reward is clipped to its sign value {-1, 0, 1}.
            print_epochs (bool):       If True, the method prints out the statistical data at the end of each epoch.
            log_episodes (bool):       If True, the method stores the information about each completed episode to a
                                       text file.
        """
        # Check the input parameters.
        assert training_steps > self.WINDOW_SIZE + 1
        assert memory_size > 0
        assert warm_up_steps > self.WINDOW_SIZE
        assert warm_up_steps <= training_steps
        assert target_update_steps > 0
        assert model_save_steps > 0
        assert epoch_steps > 0
        assert action_duration > 0
        assert eps_min >= 0.0
        assert eps_max <= 1.0
        assert eps_min <= eps_max
        assert eps_interval > 0

        # Get the first frame by resetting the environment.
        observation = self._gym_env.reset()

        # Initialize the replay memory with the first frame being the only entry for now.
        self._replay_memory = ReplayMemory(memory_size,
            DQNAtari.WINDOW_SIZE, self.process_observation(observation))

        # Initialize local variables.
        action = 0
        steps = 0
        episode_steps = 0
        episode_count = 0
        episode_eps = 0
        epoch_eps = 0
        episode_reward = 0
        epoch_reward = 0

        # If episodes are logged, print out the file header.
        if log_episodes:
            with open(self._name + "-episodes.txt", 'a') as f:
                print("steps eps reward", file=f)
                f.close()

        # Start the training loop.
        while steps < training_steps:
            new_action = steps % action_duration == 0 # Is it time to start a new action?
            warming_up = steps < warm_up_steps # Are we still just warming up?
            
            # If time to select a new action, compute the current eps and chose a random
            # action with the eps probability. While warming up, all actions are chosen
            # randomly, since the policy network is untrained and random choice is faster.
            if new_action:
                if steps >= eps_interval:
                    eps = eps_min
                else:
                    eps = (1 - steps/eps_interval) * (eps_max - eps_min) + eps_min
                episode_eps += eps
                epoch_eps += eps

                action = None

                if not warming_up:
                    if random.random() > eps:
                        action = self.get_policy_action()
                if action == None:
                    action = random.randint(0, self._num_actions - 1)

            # Execute the action and obtain its effects.
            observation, reward, done, _ = self._gym_env.step(action)

            # If the reward should be clipped, take only its sign.
            if clip_reward and reward != 0:
                reward = reward / abs(reward)
            episode_reward += reward
            
            # Add the effects of the executed action into the replay memory as an experience.
            self._replay_memory.add_experience(
                action, reward, self.process_observation(observation), done
            )

            # If not warming up, train the policy network.
            if not warming_up:
                if new_action:
                    self._train_on_batch()

                if steps % target_update_steps == 0:
                    self._update_target_network()
                
                if steps % model_save_steps == 0:
                    self.save_model(steps)

            # At the end of each epoch print out some statistics.
            if steps > 0 and epoch_steps > 0 and episode_count > 0 and steps % epoch_steps == 0:
                print("{} steps, {} episodes, avg. eps: {:.4f}, avg. reward: {:.4f}".format(
                    steps, episode_count, epoch_eps/epoch_steps, epoch_reward/episode_count))
                epoch_eps = 0
                epoch_reward = 0
                episode_count = 0

            # If episode is finished, print out some statistics and reset the environment.
            if done:
                episode_count += 1
                epoch_reward += episode_reward

                print("{} steps, {} episodes, avg. eps: {:.4f}, avg. reward: {:.4f}\r".format(
                    steps, episode_count, episode_eps/episode_steps, epoch_reward/episode_count), end="")

                if log_episodes:
                    with open(self._name + "-episodes.txt", 'a') as f:
                        print(steps, eps, episode_reward, file=f)
                        f.close()

                episode_steps = 0
                episode_eps = 0
                episode_reward = 0
                self._gym_env.reset()
            
            # Next step
            episode_steps += 1
            steps += 1

        # Save the final model.
        self.save_model(steps)

        # Free the replay memory.
        self._replay_memory = None
    
    def _train_on_batch(self, batch_size=32, gamma=0.99):
        """
        Trains the policy network on a batch of the given size of randomly chosen experiences.

        The algorithm is as follows:

        1. Choose a random batch of experiences (window0, action, reward, window1, terminal)
           from the replay memory.
        2. For each batch instance do:
               2.1 Get the policy Q-values:
                   window0 -> policy network -> policy-Q-values
               2.2 Get the target Q-values:
                   window1 -> target network -> target-Q-values
               2.3 Compute the desired Q value for the action:
                       if terminal:
                           policy-Q-values[action] <- reward
                       else:
                           policy-Q-values[action] <- reward + gamma * max(target-Q-values)
               2.4 Put the new Q-policy-values into a training batch.
        3. Train the policy network on the training batch.

        Parameters:
            batch_size (int): The size of the batch.
            gamma (float):    The future reward discount rate.
        """
        assert self._replay_memory.experience_count() >= batch_size
        
        # Randomly select a batch.
        experiences = self._replay_memory.get_random_experiences(batch_size)

        # Unpack individual batch values.
        batch_window0 = []
        batch_action = []
        batch_reward = []
        batch_window1 = []
        batch_continues = []
        i = 1
        for experience in experiences:
            (window0, action, reward, window1, terminal) = experience
            batch_window0.append(window0)
            batch_action.append(action)
            batch_reward.append(reward)
            batch_window1.append(window1)
            batch_continues.append(0.0 if terminal else 1.0)
        batch_window0 = np.array(batch_window0).astype('float')/255.0
        batch_window1 = np.array(batch_window1).astype('float')/255.0
        batch_reward = np.array(batch_reward)
        batch_continues = np.array(batch_continues)

        # Compute the policy Q-values for the batch.
        batch_policy_y = self._policy_network.predict_on_batch(batch_window0)

        # Compute the target Q-values for the batch.
        batch_target_y = self._target_network.predict_on_batch(batch_window1)

        # Compute the Q training labels values for the batch.
        max_q = np.amax(batch_target_y, axis=1) # Get the maximum target Q for each instance.
        batch_q = batch_reward + gamma * batch_continues * max_q # Compute the Q value for each instance.

        # Prepare the training batch.
        for i in range(batch_size):
            batch_policy_y[i][batch_action[i]] = batch_q[i]

        # Train the policy network on the training batch.
        self._policy_network.train_on_batch(x=batch_window0, y=batch_policy_y)

    def _update_target_network(self):
        """
        Copies the weights from the policy network to the target network.
        """
        self._target_network.set_weights(self._policy_network.get_weights())

    def test(self, testing_steps=100000, epoch_steps=10000, eps=0.1):
        """
        Test the policy network for the given number of steps that span over the
        given number of epochs.

        Parameters:
            testing_steps (int): The number of steps to test the model.
            epoch_steps (int):   The length of one epoch. At the end of each epoch
                                 the average epoch award is reported.
            eps (float):         The percentage of randomly chosen actions.
        """
        # Check the input parameters.
        assert testing_steps > self.WINDOW_SIZE
        assert epoch_steps > 0
        assert eps >= 0.0
        assert eps <= 1.0

        steps = 0
        episode_count = 0
        episode_reward = 0
        epoch_reward = 0

        # Construct a window from consequent frames.
        frame = self.process_observation(self._gym_env.reset())
        window = [frame]

        # Play the game for the given number of steps.
        while (steps < testing_steps):
            # Choose the policy or random action.
            if random.random() < eps or len(window) < DQNAtari.WINDOW_SIZE:
                action = random.randint(0, self._num_actions - 1)
            else:
                action = self.get_policy_action(window)
            
            # Execute the chosen action and obtain its effects.
            observation, reward, done, _ = self._gym_env.step(action)
            frame = self.process_observation(observation)
            episode_reward += reward

            steps += 1

            # At the end of each epoch print out some statistics.
            if steps > 0 and episode_count > 0 and steps % epoch_steps == 0:
                print("{} steps, {} episodes, avg. reward: {:.4f}".format(
                    steps, episode_count, epoch_reward/episode_count))
                epoch_reward = 0
                episode_count = 0

            # If episode is finished, print out some statistics and reset the environment.
            if done:
                episode_count += 1
                epoch_reward += episode_reward
                episode_reward = 0

                print("{} steps, {} episodes, avg. reward: {:.4f}\r".format(
                    steps, episode_count, epoch_reward/episode_count), end="")
                
                frame = self.process_observation(self._gym_env.reset())
                window = []
            
            # Construct a window from subsequent frames.
            if len(window) < DQNAtari.WINDOW_SIZE:
                # If window is not full, append the frame.
                window.append(frame)
            else:
                # If window is full, shift the frames.
                for i in range(DQNAtari.WINDOW_SIZE - 1):
                    window[i] = window[i+1]
                window[DQNAtari.WINDOW_SIZE - 1] = frame

    def play(self, episodes=1, eps=0.1, fps=30, save_frames=False):
        """
        Plays the game for the given number of episodes using the policy network. The gameplay
        is rendered in real playing time. Individual frames can be saved as .png images, so
        a video can be produced off-line (e.g. via FFmpeg).

        Parameters:
            episodes (int):     The number of episodes to play the game.
            eps (float):        The percentage of randomly chosen actions.
            fps (int):          The speed of the gameplay.
            save_frames (bool): If True, frames are saved as .png images.
        """
        # Check the input parameters.
        assert episodes >= 1
        assert eps >= 0.0
        assert eps <= 1.0
        assert fps >= 1

        frame_count = 0
        episode_count = 0
        filename = None

        # Construct a window from consequent frames.
        frame = self.process_observation(self._gym_env.reset())
        window = [frame]
        
        timer = GameTimer(fps=fps)

        # Play the game for the given number of episodes.
        while (episode_count < episodes):
            frame_count += 1

            # Save the frame as a .png image.
            if save_frames:
                filename = self._name + "-frame" + str(frame_count) + ".png"

            # Choose the policy or random action.
            if random.random() < eps or len(window) < DQNAtari.WINDOW_SIZE:
                action = random.randint(0, self._num_actions - 1)
            else:
                action = self.get_policy_action(window)
            
            # Execute the chosen action and obtain its effects.
            observation, _, done, _ = self._gym_env.step(action)
            frame = self.process_observation(observation, filename)

            # If episode is finished, reset the environment and clear the window.
            if done:
                frame = self.process_observation(self._gym_env.reset())
                window = []
                episode_count += 1

            # Construct a window from subsequent frames.
            if len(window) < DQNAtari.WINDOW_SIZE:
                # If window is not full, append the frame.
                window.append(frame)
            else:
                # If window is full, shift the frames.
                for i in range(DQNAtari.WINDOW_SIZE - 1):
                    window[i] = window[i+1]
                window[DQNAtari.WINDOW_SIZE - 1] = frame
            
            # Sleep until the time to render the next frame.
            timer.wait_next_frame()
