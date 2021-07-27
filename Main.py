import Yooter
import numpy as np  # for array stuff and random
from PIL import Image  # for creating visual of our env
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle  # to save/load Q-Tables
from matplotlib import style  # to make pretty charts because it matters.
import time

EPISODES = 50
MOVE_PENALTY = 1  # feel free to tinker with these!
ENEMY_PENALTY = 3000  # feel free to tinker with these!
KILL_REWARD = 250  # feel free to tinker with these!
epsilon = 0.5  # randomness
EPS_END = EPISODES//2
EPS_DECAY = epsilon / EPS_END
SHOW_EVERY = 1000  # how often to play through env visually.

start_q_table = None  # if we have a pickled Q table, we'll put the filename of it here.

LEARNING_RATE = 0.1
DISCOUNT = 0.95
x = 1100
y = 1000
x = x // 10
y = y // 10
episode_rewards = []
q_table = {}
for i in range(x):
    for j in range(y):
        q_table[(i, j)] = [np.random.uniform(-5, 0) for i in range(4)]

for episode in range(EPISODES):
    game = Yooter.Game_Class()
    episode_reward = 0
    print(f"on #{episode}, epsilon is {epsilon}")
    print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
    if epsilon > 0:
        epsilon -= EPS_DECAY
    while game.running:
        game.step()

