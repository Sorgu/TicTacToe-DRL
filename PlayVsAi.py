from TicTacToe import TicTacToe
import tensorflow as tf
import numpy as np

env = TicTacToe()
game_over = False
model = tf.keras.models.load_model("model1.12.keras")
state = env.reset()
while not game_over:
    if env.current_player == 1:
        q_values = model.predict(state.reshape(1, -1))
        action = np.argmax(q_values)
        state = env.move(action)
        print(q_values)
        env.print_board()
    else:
        action = int(input("Your turn: "))
        state = env.move(action)
        env.print_board()
