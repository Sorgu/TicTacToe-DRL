import csv
import os
import random
import numpy as np
import tensorflow as tf

from Dqn import DQN
from TicTacToe import TicTacToe

def play_game_pve(epsilon, model, env):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.choice([i for i in range(9)])
        else:
            q_values = model.model.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(q_values[0])
        next_state, reward, done = env.step(action)
        model.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    return total_reward

def play_game_pvp(epsilon, model, env):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        cur_player = env.current_player
        if random.uniform(0, 1) < epsilon:
            action = random.choice([i for i in range(9)])
        else:
            correct_state = state if cur_player == 1 else state.copy()*-1
            q_values = model.model.predict(correct_state.reshape(1, -1), verbose=0)
            action = np.argmax(q_values[0])
        next_state, reward, done = env.step_pvp(action)
        model.remember(state.copy()*cur_player, action, reward, next_state.copy()*cur_player, done)
        if done and reward != -30:
            model.remember(last_state.copy()*-1*cur_player, last_action, -reward, state.copy()*-1*cur_player, done)
        state = next_state
        total_reward += reward
        last_state = state
        last_action = action
    return total_reward

def train_dqn(episodes, file_name, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, batch_size=64, epochs=10, pvp=False):
    env = TicTacToe()
    model = DQN()
    target_model = DQN()
    target_model.set_weights(model.get_weights())
    total_losses = []
    loss_file_name = f"loss-history/loss-{file_name}.csv"
    model_file_name = f"{file_name}.keras"
    # Creates loss file if it does not exist
    if not os.path.exists(loss_file_name):
        with open(loss_file_name, "w") as loss_file:
            pass
    # Loads the model if it exists already
    if os.path.exists(model_file_name):
        model.model = tf.keras.models.load_model(model_file_name)
        target_model.model = tf.keras.models.load_model(model_file_name)


    for episode in range(episodes):
        if pvp:
            total_reward = play_game_pvp(epsilon, model, env)
        else:
            total_reward = play_game_pve(epsilon, model, env)
        replay_losses = model.replay(epochs, batch_size, target_model, smart_predict=-1)
        for x in replay_losses:
            total_losses.append(x)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % 100 == 0:
            target_model.set_weights(model.get_weights())
            with open(loss_file_name, "a") as loss_file:
                writer = csv.writer(loss_file)
                writer.writerow(total_losses)
                total_losses = []
            model.model.save(model_file_name)

        print(f"Episode {episode}/{episodes}, Reward: {total_reward}, Epsilon: {epsilon}")
    with open(loss_file_name, "a") as loss_file:
        writer = csv.writer(loss_file)
        writer.writerow(total_losses)
    model.model.save(model_file_name)

train_dqn(20000, "model1.12", pvp=True, epsilon=0.1)
