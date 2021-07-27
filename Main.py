import Yooter
import numpy as np  # for array stuff and random
import math
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
from matplotlib import style  # to make pretty charts because it matters.
import time

EPISODES = 50000
MOVE_PENALTY = 1  # feel free to tinker with these!
ENEMY_PENALTY = 3000  # feel free to tinker with these!
KILL_REWARD = 25000  # feel free to tinker with these!
epsilon = 0.25  # randomness
EPS_END = EPISODES//4
EPS_DECAY = epsilon / EPS_END
SHOW_EVERY = 1000  # how often to play through env visually.

start_q_table = None  # if we have a pickled Q table, we'll put the filename of it here.

LEARNING_RATE = 0.1
DISCOUNT = 0.95
x = 1100
y = 1000
x = x // 25
y = y // 25
episode_rewards = []
q_table = {}
for i in range(-x, x):
    for j in range(-y, y):
        for k in range(0, 359,12):
            q_table[(i, j, k)] = [np.random.uniform(-5, 0) for i in range(12)]

for episode in range(EPISODES):
    time_running = 0
    if episode == 12537:
        input()
    if episode % SHOW_EVERY == 0:
        game = Yooter.Game_Class(True)
    else:
        game = Yooter.Game_Class(False)
    episode_reward = 0
    reward = 0
    print(f"on #{episode}, epsilon is {epsilon}")
    if epsilon > 0:
        epsilon -= EPS_DECAY
    game.step()
    game.step()
    while game.running:
        closest_enemy = [9999999, (0, 0, game.rotation % 360)]
        if len(game.enemyList) != 0:
            for enemy in game.enemyList:
                distance = math.sqrt((enemy.x - 400) ** 2 + (enemy.y - 300) ** 2)
                if distance < closest_enemy[0]:
                    closest_enemy[0] = distance
                    closest_enemy[1] = (enemy.x, enemy.y, game.rotation % 360)

            obs = (closest_enemy[1][0] // 25, closest_enemy[1][1] // 25, game.rotation % 360)
            try:
                if np.random.random() > epsilon:
                    game.action = np.argmax(q_table[obs])
                else:
                    game.action = np.random.randint(0, 12)
            except:
                game.action = 0
            game.step()
            if len(game.enemyList) == 0:
                game.step()

            if time_running > 100000:
                game.running = False


            if game.action == 11:
                reward = 0
            else:
                reward = -MOVE_PENALTY
            if game.enemies_killed != 0:
                reward = KILL_REWARD
                game.enemies_killed = 0
            if game.running == False:
                reward = -ENEMY_PENALTY

            try:
                new_obs = (closest_enemy[1][0]//25, closest_enemy[1][1]//25, game.rotation % 360)  # new observation
                max_future_q = np.max(q_table[new_obs])  # max Q value for this new obs
                current_q = q_table[obs][game.action]  # current Q for our chosen action
                if reward == KILL_REWARD:
                    new_q = KILL_REWARD
                    time_running = 0
                elif reward == -ENEMY_PENALTY:
                    new_q = -ENEMY_PENALTY
                else:
                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[new_obs][game.action] = new_q
            except:
                pass
            '''if 15 > game.player_one.position_x - closest_enemy[1][0] > -15:
                if 15 > game.player_one.position_y - closest_enemy[1][1] > -15:

                    print("player:", game.player_one.position_x, game.player_one.position_y)
                    print("enemy:", closest_enemy[1][0], closest_enemy[1][1])'''







