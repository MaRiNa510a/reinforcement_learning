import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from PIL import Image

im = cv2.imread("smile_test.png")
cv2.imshow("input image", im)

style.use("ggplot")

h, w, _ = im.shape
SIZE_x = w
SIZE_y = h

HM_EPISODES = 100

MOVE_PENALTY = 1
BLACK_REWARD = 25
WHITE_PENALTY = 300

epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000

start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

AGENT_N = 1
BLACK_N = 2
WHITE_N = 3

d = {
    1:(0, 0, 255),       #Red
    2:(0, 0, 0),         #Black
    3:(255, 255, 255)    #White
    }


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE_x)
        self.y = np.random.randint(0, SIZE_y)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y) #?

    def action(self, choice):
        #up
        if choice == 0:
            self.move(x=0, y=1)
        #down
        elif choice == 1:
            self.move(x=0, y=-1)
        #left
        elif choice == 2:
            self.move(x=-1, y=0)
        #right
        elif choice == 3:
            self.move(x=1, y=0)

    def move(self, x=False, y=False):
        # x=False
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        # y=False
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE_x -1:
            self.x = SIZE_x -1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE_y -1:
            self.y = SIZE_y -1

if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE_x+1, SIZE_x):
        for y1 in range(-SIZE_y+1, SIZE_y):
            for x2 in range(-SIZE_x+1, SIZE_x):
                for y2 in range(-SIZE_y+1, SIZE_y):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f: #rb means read the binary file
        q_table = pickle.load(f)

for episode in range(HM_EPISODES):
    agent = Blob()
    black_point = Blob()
    white_point = Blob()

    episode_reward = []
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon:{epsilon}")
        #np.mean shows the average of elements each episode
        #episode_rewards[-SHOW_EVERY:] means picking up character string from -SHOW_EVERY to last
        # print(f"{SHOW_EVERY} episode reward average {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    # episode_reward = []
    for i in range(200):
        #observation
        obs = (agent - black_point, agent - white_point)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

    agent.action(action)

    if agent.x == black_point.x and agent.y == black_point.y:
        reward = BLACK_REWARD
    elif agent.x == white_point.x and agent.y == white_point.y:
        reward = -WHITE_PENALTY
    else:
        reward = -MOVE_PENALTY

    new_obs = (agent - black_point, agent - white_point)
    max_future_q = np.max(q_table[new_obs])
    current_q = q_table[obs][action]

    if reward == BLACK_REWARD:
        new_q = BLACK_REWARD
    elif reward == -WHITE_PENALTY:
        new_q = -WHITE_PENALTY
    else:
        new_q =(1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

    q_table[obs][action] = new_q

    if show:
        env = np.zeros((SIZE_x, SIZE_y, 3), dtype = np.uint8)  #3 means BGR
        env[agent.y][agent.x] = d[AGENT_N]
        env[black_point.y][black_point.x] = d[BLACK_REWARD]
        env[white_point.y][white_point.x] = d[WHITE_PENALTY]

        img = Image.fromarray(env, "RGB")
        img = img.resize((300, 300))
        cv2.imshow("", np.array(img))
        if reward == BLACK_REWARD or reward == WHITE_PENALTY:
            if cv2.waitKey(500) & 0xFF == ord("q"):
                break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    episode_reward.append(episode_reward)
    epsilon *= EPS_DECAY


