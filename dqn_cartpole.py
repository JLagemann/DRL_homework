import random
import gym
import math
import numpy as np
import time
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class TheBrain(tf.keras.Model):
    def __init__(self, output_units=2):
        super(TheBrain, self).__init__()
        self.layer_1 = Dense(24, input_dim=4, activation='tanh')
        self.layer_2 = Dense(48, activation='tanh')
        self.layer_3 = Dense(output_units, activation='linear')
    
    def call(self, input):
        x = self.layer_1(input)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x


class DQNBalancer():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=0.05, alpha=0.01, batch_size=64, quiet=False):
        self.replay_buffer = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        # Init model
        self.model = TheBrain(2)
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.replay_buffer, min(len(self.replay_buffer), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            if done:
                y_target[0][action] = reward
            else:
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print(f'Ran {e} episodes. Solved after {e - 100} trials')
                return e - 100
            if e % 100 == 0 and not self.quiet:
                print(f'[Episode {e}] - Mean survival time over last 100 episodes was {mean_score} ticks.')

            self.replay(self.batch_size)
        
        if not self.quiet: print(f'Failed to solve after {e} episodes')
        return e

if __name__ == '__main__':
    agent = DQNBalancer()
    agent.run()