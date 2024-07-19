from TicTacToe import TicTacToe
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("model1.3.keras")
player1_wins = 0
player2_wins = 0
draws = 0
illegal_moves = 0
games_played = 1000
for i in range(games_played):
    print(f"{i}/{games_played}")
    env = TicTacToe()
    state = env.reset()
    game_over = False
    while not game_over:
        if env.current_player == 1:
            q_values = model.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(q_values)
            if env.board.flatten()[action] != 0:
                game_over = True
                illegal_moves += 1
                env.print_board()
                print(action)
            state = env.move(action)
            if env.check_winner(1):
                game_over = True
                player1_wins += 1
            elif env.check_draw():
                game_over = True
                draws += 1
        else:
            state = env.move(env.rand_legal_move())
            if env.check_winner(-1):
                game_over = True
                player2_wins += 1
            elif env.check_draw():
                game_over = True
                draws += 1
print(f"Player 1 wins: {player1_wins}\nPlayer 2 wins: {player2_wins}\nDraws: {draws}\nIllegal moves: {illegal_moves}")