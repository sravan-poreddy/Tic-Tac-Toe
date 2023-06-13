from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)

# Q-learning parameters
alpha = 0.5  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate

# Initialize Q-table
q_table = {}


def get_state(board):
    # Convert the 3x3 board into a unique string representation
    return str(board.flatten())


def get_valid_moves(board):
    # Return the indices of all empty cells on the board
    return np.argwhere(board.flatten() == 0).flatten()


def get_best_move(board):
    # Get the best move from the Q-table for the current state
    state = get_state(board)
    valid_moves = get_valid_moves(board)

    if state not in q_table:
        return random.choice(valid_moves)

    q_values = q_table[state]
    best_moves = np.where(q_values == np.max(q_values))[0]
    return random.choice(best_moves)


def update_q_table(state, action, reward, next_state):
    # Update the Q-table based on the Q-learning formula
    if state not in q_table:
        q_table[state] = np.zeros(9)

    q_values = q_table[state]
    next_q_values = q_table[next_state] if next_state in q_table else np.zeros(9)
    q_values[action] = (1 - alpha) * q_values[action] + alpha * (reward + gamma * np.max(next_q_values))


def check_winner(board):
    # Check if there is a winner
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]  # diagonals
    ]

    for combination in winning_combinations:
        if board[combination[0]] == board[combination[1]] == board[combination[2]] != 0:
            return board[combination[0]]

    if 0 not in board:
        return 0  # draw

    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/move', methods=['POST'])
def move():
    board = np.array(request.json['board'])

    # Check if the game is already over
    winner = check_winner(board)
    if winner is not None:
        return jsonify({'move': -1, 'winner': int(winner)})

    # Choose a move
    if random.uniform(0, 1) < epsilon:
        # Explore: choose a random move
        valid_moves = get_valid_moves(board)
        move = random.choice(valid_moves)
    else:
        # Exploit: choose the best move
        move = get_best_move(board)

    # Make the move and check the result
    new_board = np.copy(board)
    new_board[move] = 1

    winner = check_winner(new_board)
    if winner is not None:
        return jsonify({'move': int(move), 'winner': int(winner)})

    # Opponent's move
    opponent_move = get_best_move(new_board)
    new_board[opponent_move] = -1

    # Check the result after the opponent's move
    winner = check_winner(new_board)
    if winner is not None:
        return jsonify({'move': int(move), 'opponent_move': int(opponent_move), 'winner': int(winner)})

    # Update Q-table
    state = get_state(board)
    next_state = get_state(new_board)
    reward = 0  # no immediate reward
    update_q_table(state, move, reward, next_state)

    return jsonify({'move': int(move), 'opponent_move': int(opponent_move), 'winner': None})


if __name__ == '__main__':
    app.run()
