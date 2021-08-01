import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer
from keras.optimizers import adam_v2
from collections import deque
import time
import random
import Yooter
import matplotlib.pyplot as plt

POSSIBLE_ACTIONS = 12

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MEMORY_FRACTION = 0.20
KILL_REWARD = 5
MOVEMENT_PENALTY = -1
DEATH_PENALTY = -6

# Environment settings
EPISODES = 5000
SHOW_EVERY = 100

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = epsilon / (EPISODES // 2)

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes


class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(InputLayer(input_shape=(100, 2)))

        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Flatten())
        model.add(Dense(POSSIBLE_ACTIONS, activation='softmax'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=adam_v2.Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def get_action(self, state):
        action = self.model.predict(state.reshape(-1, *state.shape))
        return np.argmax(action)

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        '''currrent_qs_list = []
        for state in current_states:
            pre = self.target_model.predict(state.reshape(-1, *state.shape))
            currrent_qs_list.append(pre[0])
        current_qs_list = np.array(currrent_qs_list)'''

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        '''future_qs_list = []
        for state in new_current_states:
            pre = self.target_model.predict(state.reshape(-1, *state.shape))
            future_qs_list.append(pre[0])
        future_qs_list = np.array(future_qs_list)'''


        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


agent = DQNAgent()
episode_rewards = []
state_shape = []
for _ in range(100):
    state_shape.append([0, 0])

state_shape = np.array(state_shape)

for episode in range(EPISODES):
    time_without_kill = 0
    if EPISODES == 1:
        game = Yooter.Game_Class(True)
    else:
        game = Yooter.Game_Class(False)
    game.step()

    episode_reward = 0

    print(f'episode: {episode+1}/{EPISODES}, epsilon: {epsilon}')

    while game.running:
        reward = 0
        current_state = state_shape.copy()
        current_state[0][0] = game.rotation%360
        if len(game.enemyList) == 0:
            game.step()
        j = 1
        for enemy in game.enemyList:
            try:
                current_state[j][0] = enemy.x
                current_state[j][1] = enemy.y
            except Exception as e:
                print(e)
            j += 1

        if random.random() > epsilon:
            game.action = agent.get_action(current_state)
        else:
            game.action = random.randint(0, 11)

        game.step()

        new_state = state_shape.copy()
        new_state[0][0] = game.rotation % 360
        j = 1
        for enemy in game.enemyList:
            try:
                new_state[j][0] = enemy.x
                new_state[j][1] = enemy.y
            except:
                pass
            j += 1

        if time_without_kill >= 1000:
            game.running = False

        if not game.running:
            reward = DEATH_PENALTY
        elif game.enemies_killed > 0:
            time_without_kill = 0
            reward = KILL_REWARD
        else:
            reward = MOVEMENT_PENALTY
        time_without_kill += 1
        episode_reward += reward

        agent.update_replay_memory((current_state, game.action, reward, new_state, game.running == False))
        agent.train(game.running == False)

    episode_rewards.append(episode_reward)
    if epsilon > 0:
        epsilon -= EPSILON_DECAY
max_reward = max(episode_rewards)
min_reward = min(episode_rewards)
average_reward = sum(episode_rewards) // len(episode_rewards)
agent.model.save(
    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
