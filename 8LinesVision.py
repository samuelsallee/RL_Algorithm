import Yooter
import numpy as np  # for array stuff and random
import math
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import time
import pickle


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def onSegment(p, q, r):
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p, q, r):
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:
        return 1
    elif val < 0:
        return 2
    else:
        return 0


def doIntersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if (o1 != o2) and (o3 != o4):
        return True

    if (o1 == 0) and onSegment(p1, p2, q1):
        return True

    if (o2 == 0) and onSegment(p1, q2, q1):
        return True

    if (o3 == 0) and onSegment(p2, p1, q2):
        return True

    if (o4 == 0) and onSegment(p2, q1, q2):
        return True

    return False


def intersects(enemy, line):
    linep = Point(line[0][0], line[0][1])
    lineq = Point(line[1][0], line[1][1])
    x = enemy.x - 32
    y = enemy.y - 32
    enemyp1 = Point(x, y)
    enemyp2 = Point(x, y + 64)
    enemyp3 = Point(x + 64, y)
    enemyp4 = Point(x + 64, y + 64)

    return doIntersect(linep, lineq, enemyp1, enemyp2) or \
           doIntersect(linep, lineq, enemyp1, enemyp3) or \
           doIntersect(linep, lineq, enemyp4, enemyp2) or \
           doIntersect(linep, lineq, enemyp4, enemyp3)


def get_distance(enemy):
    return math.sqrt((enemy.x - 400) ** 2 + (enemy.y - 300) ** 2)


EPISODES = 1
MOVE_PENALTY = .00001  # feel free to tinker with these!
ENEMY_PENALTY = 6  # feel free to tinker with these!
KILL_REWARD = 5000  # feel free to tinker with these!
if EPISODES == 1:
    epsilon = 0
    EPS_END = 1
else:
    epsilon = .5  # randomness
    EPS_END = EPISODES // 2
EPS_DECAY = epsilon / EPS_END
SHOW_EVERY = 100  # how often to play through env visually.
LEARNING_RATE = 0.1
DISCOUNT = 0.95
episode_rewards = []
eight_lines = [((400, 300), (400, 600)), ((400, 300), (400, 0)), ((400, 300), (100, 300)), ((400, 300), (700, 300)),
               ((400, 300), (100, 0)), ((400, 300), (700, 600)), ((400, 300), (100, 600)), ((400, 300), (700, 0))]

q_table = {}
start_q_table = '8qtable-1627647844.pickle'  # if we have a pickled Q table, we'll put the filename of it here.

if start_q_table is None:
    for line in eight_lines:
        for line2 in eight_lines:
            if line2 != line:
                for line3 in eight_lines:
                    if line3 != line2 and line3 != line:
                        for line4 in eight_lines:
                            if line4 != line3 and line4 != line2 and line4 != line:
                                for line5 in eight_lines:
                                    if line5 != line4 and line5 != line3 and line5 != line2 and line5 != line:
                                        for k in range(0, 359, 12):
                                            q_table[(line, line2, line3, line4, line5, k)] = [np.random.uniform(-5, 0) for i in range(8)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

for episode in range(EPISODES):
    time_running = 0
    if EPISODES == 1:
        game = Yooter.Game_Class(True, True)
        game.step()
    else:
        game = Yooter.Game_Class(False)
        game.step()

    episode_reward = 0
    reward = 0
    print(f"on #{episode}, epsilon is {epsilon}")
    if epsilon > 0:
        epsilon -= EPS_DECAY

    while game.running:
        if len(game.enemyList) == 0:
            game.step()


        closest_enemies_and_lines = [[300.0, ((400, 300), (400, 600))], [300.0, ((400, 300), (400, 000))],
                                     [300.0, ((400, 300), (100, 300))], [300.0, ((400, 300), (700, 300))],
                                     [425.0, ((400, 300), (100, 000))], [425.0, ((400, 300), (700, 600))],
                                     [425.0, ((400, 300), (100, 600))], [425.0, ((400, 300), (700, 000))]]
        for enemy in game.enemyList:
            for i in range(8):
                if intersects(enemy, eight_lines[i]):
                    enemy_distance = get_distance(enemy)
                    if closest_enemies_and_lines[i][0] > enemy_distance:
                        closest_enemies_and_lines[i][0] = enemy_distance

        closest_enemies_and_lines.sort(key=lambda x: x[0])
        obs = (closest_enemies_and_lines[0][1], closest_enemies_and_lines[1][1], closest_enemies_and_lines[2][1],
               closest_enemies_and_lines[3][1], closest_enemies_and_lines[4][1],game.rotation)
        try:
            if np.random.random() > epsilon:
                game.action = np.argmax(q_table[obs])
            else:
                game.action = np.random.randint(0, 8)
        except:
            pass

        game.step()

        if time_running > 500:
            game.running = False
        else:
            time_running += 1

        reward = -MOVE_PENALTY
        if game.enemies_killed > 0:
            reward = KILL_REWARD
            game.enemies_killed = 0
            time_running = 0
        if not game.running:
            reward = -ENEMY_PENALTY

        try:
            if len(game.enemyList) == 0:
                game.step()

            closest_enemies_and_lines = [[300.0, ((400, 300), (400, 600))], [300.0, ((400, 300), (400, 000))],
                                         [300.0, ((400, 300), (100, 300))], [300.0, ((400, 300), (700, 300))],
                                         [425.0, ((400, 300), (100, 000))], [425.0, ((400, 300), (700, 600))],
                                         [425.0, ((400, 300), (100, 600))], [425.0, ((400, 300), (700, 000))]]
            for enemy in game.enemyList:
                for i in range(8):
                    if intersects(enemy, eight_lines[i]):
                        enemy_distance = get_distance(enemy)
                        if closest_enemies_and_lines[i][0] > enemy_distance:
                            closest_enemies_and_lines[i][0] = enemy_distance

            closest_enemies_and_lines.sort(key=lambda x: x[0])
            new_obs = (closest_enemies_and_lines[0][1], closest_enemies_and_lines[1][1],
                       closest_enemies_and_lines[2][1], closest_enemies_and_lines[3][1],
                       closest_enemies_and_lines[4][1],game.rotation)

            max_future_q = np.max(q_table[new_obs])  # max Q value for this new obs
            current_q = q_table[obs][game.action]  # current Q for our chosen action
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

if EPISODES != 1:
    with open(f"8qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)
