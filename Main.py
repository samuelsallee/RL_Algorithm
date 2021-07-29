import Yooter
import numpy as np  # for array stuff and random
import math
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import time
import pickle

EPISODES = 1
MOVE_PENALTY = .00001  # feel free to tinker with these!
ENEMY_PENALTY = 6  # feel free to tinker with these!
KILL_REWARD = 5000  # feel free to tinker with these!
epsilon = 0  # randomness
EPS_END = 1 #EPISODES // 2
EPS_DECAY = epsilon / EPS_END
SHOW_EVERY = 100  # how often to play through env visually.

start_q_table = 'qtable-1627566717.pickle'  # if we have a pickled Q table, we'll put the filename of it here.

LEARNING_RATE = 0.1
DISCOUNT = 0.95
x = 400
y = 300
x = x // 25
y = y // 25
episode_rewards = []
q_table = {}
if start_q_table is None:
    for i in range(32):
        for j in range(24):
            for k in range(0, 359, 12):
                q_table[(i, j, k)] = [np.random.uniform(-5, 0) for i in range(8)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


for episode in range(EPISODES):
    time_running = 0
    if EPISODES == 1:
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
        else:
            game.step()
            for enemy in game.enemyList:
                distance = math.sqrt((enemy.x - 400) ** 2 + (enemy.y - 300) ** 2)
                if distance < closest_enemy[0]:
                    closest_enemy[0] = distance
                    closest_enemy[1] = (enemy.x, enemy.y, game.rotation % 360)

        obs = (int(closest_enemy[1][0] // 25), int(closest_enemy[1][1] // 25), game.rotation % 360)
        try:
            if np.random.random() > epsilon:
                game.action = np.argmax(q_table[obs])
            else:
                game.action = np.random.randint(0, 8)
        except:
            game.action = 0
        game.step()
        if time_running > 500:
            game.running = False
        else:
            time_running += 1

        reward = -MOVE_PENALTY
        if game.enemies_killed > 0:
            reward = KILL_REWARD
            game.enemies_killed = 0
        if not game.running:
            reward = -ENEMY_PENALTY

        try:
            new_obs = (int(closest_enemy[1][0] // 25), int(closest_enemy[1][1] // 25), game.rotation % 360)  # new observation
            max_future_q = np.max(q_table[new_obs])  # max Q value for this new obs
            current_q = q_table[obs][game.action]  # current Q for our chosen action
            if reward == KILL_REWARD:
                new_q = KILL_REWARD
                time_running = 0
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[obs][game.action] = new_q
        except:
            pass
        episode_reward += reward
    episode_rewards.append(episode_reward)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
print(q_table)

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)