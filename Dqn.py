import random
import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque

class DQN:
    def __init__(self):
        self.model = Sequential([
            Dense(1024, input_dim=9, activation='relu'),
            Dense(512, activation='relu'),
            Dense(9, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.replay_buffer = deque(maxlen=2000)
        self.gamma = 0.5

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    def set_weights(self, weights):
        self.model.set_weights(weights)
    def get_weights(self):
        return self.model.get_weights()

    def replay(self, epochs, batch_size, target_model, smart_predict=-1): # smart_predict on with -1 and off with 1. Tries to predict the best move for the opponent.
        total_losses = []
        if len(self.replay_buffer) >= batch_size:
            for _ in range(epochs):
                minibatch = random.sample(self.replay_buffer, batch_size)
                states = np.array([m[0] for m in minibatch])
                actions = np.array([m[1] for m in minibatch])
                rewards = np.array([m[2] for m in minibatch])
                next_states = np.array([m[3] for m in minibatch])
                dones = np.array([m[4] for m in minibatch])

                q_values_next = target_model.model.predict(smart_predict * next_states, verbose=0)
                targets = rewards + self.gamma * smart_predict * np.amax(q_values_next, axis=1) * (1 - dones)

                q_values = self.model.predict(states, verbose=0)
                for i, action in enumerate(actions):
                    q_values[i][action] = targets[i]

                losses = self.model.train_on_batch(states, q_values)
                total_losses.append(losses)
        return total_losses
