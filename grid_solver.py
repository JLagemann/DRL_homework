import numpy as np
from collections import deque 
import gym
from gridworlds import GridWorld
# make a Q Table the size of the states * actions
# a function to choose an action
class QTab():
    def __init__(self, env, n_episodes=100, height=10, width=10, action_space=4, epsilon = 0.9, alpha = 0.1, gamma=0.95):
        self.replay_buffer = deque(maxlen=100000)
        self.n_episodes = n_episodes
        self.q_values = np.zeros((height, width, action_space))
        self. epsilon = epsilon
        self.n_episodes
        self.env = env
        self.gamma = gamma
        self.alpha = alpha

    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.randint(0,4)
        else:
            (height, width) = state
            return np.argmax(self.q_values[height, width])
    
    def update(self, state, action, new_value):
        (height, width) = state
        self.q_values[height, width, action] = (1-self.alpha) * self.q_values[height, width, action] + self.alpha * new_value

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        for state, action, reward, next_state, done in reversed(self.replay_buffer):
            if done:
                self.update(state, action, reward)
            else:
                (h,w) = next_state
                self.update(state, action, self.gamma * np.mean(self.q_values[h, w, :]))

    def run(self):
        scores = deque(maxlen=100)
        for e in range(self.n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
            self.replay()
        print(self.q_values)


if __name__ == '__main__':
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 3,
        "width": 4,
        "action_dict": action_dict,
        "start_position": (0, 0),
        "reward_position": (2, 3),
    }

    env = GridWorld(**env_kwargs)
    agent = QTab(env, height=3, width=4)
    agent.run()    


