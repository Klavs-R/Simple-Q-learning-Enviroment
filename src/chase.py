import numpy as np
from numpy import random as rnd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

style.use("ggplot")


class Bob:
    def __init__(self, size):
        """
        Class for moving "blobs" on a (size x size) board.
        :param size: board size
        """
        self.board_size = size
        self.x = rnd.randint(0, size)
        self.y = rnd.randint(0, size)

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def reset(self):
        """Reset blob to random position within the board."""
        self.x = rnd.randint(0, self.board_size)
        self.y = rnd.randint(0, self.board_size)

    def action(self, choice):
        """Choice of actions, one of 4 diagonal moves."""
        if choice == 0:
            self.move(1, 1)
        elif choice == 1:
            self.move(-1, -1)
        elif choice == 2:
            self.move(1, -1)
        elif choice == 3:
            self.move(-1, 1)

    def move(self, x=0, y=0):
        """
        Moves blob in selected manner, if action is not provided, blob will be
        moved randomly.
        """
        if not x:
            self.x += rnd.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += rnd.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > self.board_size - 1:
            self.x = self.board_size - 1

        if self.y < 0:
            self.y = 0
        elif self.y > self.board_size - 1:
            self.y = self.board_size - 1


class Env:
    def __init__(self, model="", size=10, enemy_num=1, moving=False,
                 pcol=(255, 175, 0), fcol=(0, 255, 0), ecol=(0, 0, 255)):
        """
        Creates a chase game environment. Last 3 parameters are colour
        selection for player, food and enemy respectively.
        :param model: pre-existing q-learning model (if none provided, random
                      one will be generated).
        :param size: board size
        :param enemy_num: number of enemies
        :param moving: Whether or not enemies & food move
        """

        if model:
            with open(model, 'rb') as f:
                self.qtable = pickle.load(f)
        else:
            self.qtable = {}
            for x1 in range(-size + 1, size):
                for y1 in range(-size + 1, size):
                    for x2 in range(-size + 1, size):
                        for y2 in range(-size + 1, size):
                            self.qtable[((x1, y1), (x2, y2))] = \
                                [rnd.uniform(-5, 0) for i in range(4)]

        self.size = size
        self.moving = moving
        self.pcol = pcol
        self.fcol = fcol
        self.ecol = ecol
        self.success_im = Image.new("RGB", (400, 400), (0, 150, 0))
        self.fail_im = Image.new("RGB", (400, 400), (0, 0, 150))

        self.player = Bob(self.size)
        self.food = Bob(self.size)
        self.enemies = [Bob(self.size) for i in range(enemy_num)]

    def change_enemies(self, enemy_num):
        self.enemies = [Bob(self.size) for i in range(enemy_num)]

    def make_step(env, action):
        """Takes a single time step within environment"""
        res = [0, 0]
        ppos = [env.player.x, env.player.y]
        env.player.action(action)
        if env.moving:
            env.food.move()
            for enemy in env.enemies:
                enemy.move()

        if ppos == [env.food.x, env.food.y]:
            res[0] = 1

        for enemy in env.enemies:
            if ppos == [enemy.x, enemy.y]:
                res[1] = 1

        return res

    def show(env, episodes):
        """
        Displays a visual demonstration of the agent within the game for a
        given number of episodes.
        """
        for episode in range(episodes):
            env.player.reset()
            env.food.reset()
            for enemy in env.enemies:
                enemy.reset()

            for step in range(50):
                obs = (env.player - env.food,
                       min([env.player - enemy for enemy in env.enemies]))

                env_arr = np.zeros((env.size, env.size, 3), dtype=np.uint8)
                env_arr[env.food.y][env.food.x] = env.fcol
                env_arr[env.player.y][env.player.x] = env.pcol
                for enemy in env.enemies:
                    env_arr[enemy.y][enemy.x] = env.ecol

                img = Image.fromarray(env_arr, "RGB")
                img = img.resize((400, 400))
                cv2.imshow("Get em", np.array(img))

                res = env.make_step(np.argmax(env.qtable[obs]))

                if res[0] == 1:
                    cv2.imshow("Get em", np.array(env.success_im))
                    cv2.waitKey(300)
                    break
                elif res[1] == 1:
                    cv2.imshow("Get em", np.array(env.fail_im))
                    cv2.waitKey(300)
                    break
                else:
                    if cv2.waitKey(50) & 0xFF == ord("q"):
                        break
        cv2.destroyAllWindows()

    def train(env,
              steps=300,
              mpenalty=1,
              epenalty=1000,
              freward=25,
              epsilon=0.2,
              ep_decay=0.00003,
              learning_rate=0.1,
              discount=0.95,
              episodes=1000000,
              roll_avg=5000):
        """
        Train the model associated with the environment over given number of
        episodes. Most parameters are traditional training parameters for
        q-learnign.
        :param steps: Number of steps allowed per episode
        :param episodes: Number of episodes
        :param mpenalty: Penalty per move
        :param epenalty: Penalty for colliding with enemy
        :param freward: reward for getting food
        :param roll_avg: Number of episodes to rolling average over
        :return: List of rolling averages for every episode beyond "roll_avg"
        """

        episode_rewards = []
        for episode in range(episodes):
            if episode != 0 and not episode % roll_avg:
                print(f'episode: {episode}, epsilon = {epsilon}')
                print(f'{roll_avg} ep mean: '
                      f'{np.mean(episode_rewards[-roll_avg:])}')

            env.player.reset()
            env.food.reset()
            for enemy in env.enemies:
                enemy.reset()

            episode_reward = 0
            for step in range(steps):
                obs = (env.player - env.food,
                       min([env.player - enemy for enemy in env.enemies]))

                if rnd.random() < epsilon:
                    action = rnd.randint(0, 4)
                else:
                    action = np.argmax(env.qtable[obs])

                res = env.make_step(action)
                closest = min([env.player - enemy for enemy in env.enemies])
                new_obs = (env.player - env.food, closest)
                max_future_q = np.max(env.qtable[new_obs])
                current_q = env.qtable[obs][action]

                if res[0] == 1:
                    new_q = freward
                    episode_reward += freward
                elif res[1] == 1:
                    new_q = -epenalty
                    episode_reward -= epenalty
                else:
                    episode_reward -= mpenalty
                    new_q = ((1 - learning_rate) * current_q + learning_rate *
                             (-mpenalty + discount * max_future_q))

                env.qtable[obs][action] = new_q
                if 1 in res:
                    break

            episode_rewards.append(episode_reward)
            epsilon *= 1 - ep_decay

        moving_avg = np.convolve(episode_rewards,
                                 np.ones((roll_avg,)) / roll_avg,
                                 mode='valid')
        return moving_avg

    def test(env, model, episodes, roll_avg=100,
             freward=25, mpenalty=1, epenalty=1000):
        """
        Allows for testing of a different model within current environment.
        Rewards and penalties are chosen to give meaning to the tests.
        :param model: Model being tested
        :param episodes: Number of episodes to test over
        :param roll_avg: Number of episodes to rolling average over
        :return: List of rolling averages for every episode beyond "roll_avg"
        """

        episode_rewards = []
        for episode in range(episodes):
            env.player.reset()
            env.food.reset()
            for enemy in env.enemies:
                enemy.reset()

            episode_reward = 0
            for step in range(50):
                obs = (env.player - env.food,
                       min([env.player - enemy for enemy in env.enemies]))

                res = env.make_step(np.argmax(model[obs]))

                if res[0] == 1:
                    episode_reward = freward
                    break
                elif res[1] == 1:
                    episode_reward = -epenalty
                    break
                else:
                    episode_reward -= mpenalty

            episode_rewards.append(episode_reward)

        moving_avg = np.convolve(episode_rewards,
                                 np.ones((roll_avg,)) / roll_avg,
                                 mode='valid')
        return moving_avg


if __name__ == "__main__":
    """
    Code used to generate models and training data saved in this directory
    """
    game_s = Env()
    game_m = Env(moving=True)

    avg = game_s.train()
    plt.plot(range(len(avg)), avg, label="stationary")
    with open("1M_stationary_avgs.pickle", "wb") as f:
        pickle.dump(avg, f)

    avg = game_m.train()
    plt.plot(range(len(avg)), avg, label="moving")
    with open("1M_moving_avgs.pickle", "wb") as f:
        pickle.dump(avg, f)

    plt.ylabel("reward (5000 episode rolling average)")
    plt.xlabel("episode #")
    plt.legend()
    plt.show()

    with open("1Ms_model.pickle", "wb") as f:
        pickle.dump(game_s.qtable, f)

    with open("1Mm_model.pickle", "wb") as f:
        pickle.dump(game_m.qtable, f)
