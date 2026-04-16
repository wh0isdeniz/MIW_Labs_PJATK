import random
import matplotlib.pyplot as plt

# All possible moves in the game
MOVES = ["Rock", "Paper", "Scissors"]

# Defines which move each move beats
BEATS = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper"
}

# Player 1's strategy: a Markov chain transition matrix.
# Each key is the previous move, and the value is a dict of probabilities
# for the next move. Player 1 tends to switch moves rather than repeat.
# Unlike Mr. Majewski, I sometimes make the same move twice :)
transition_matrix = {
    "Rock": {
        "Paper": 0.4,
        "Scissors": 0.4,
        "Rock": 0.2
    },
    "Paper": {
        "Scissors": 0.4,
        "Rock": 0.4,
        "Paper": 0.2
    },
    "Scissors": {
        "Rock": 0.4,
        "Paper": 0.4,
        "Scissors": 0.2
    }
}
#Another example transition matrix to see the actual importance of values
#transition_matrix = {
#    "Rock": {
#        "Paper": 0,
#        "Scissors": 0.6,
#        "Rock": 0.4
#    },
#    "Paper": {
#        "Scissors": 0.6,
#       "Rock": 0.4,
#       "Paper": 0
#   },
#   "Scissors": {
#       "Rock": 0.0,
#       "Paper": 0.6,
#       "Scissors": 0.4
#   }
#}

# Player 2's Bayesian learning structure.
# Counts how often Player 1 transitions from one move to another.
# Initialized with 1s (uniform prior) to avoid zero probabilities at the start.
learning_counts = {
    "Rock": {"Rock": 1, "Paper": 1, "Scissors": 1},
    "Paper": {"Rock": 1, "Paper": 1, "Scissors": 1},
    "Scissors": {"Rock": 1, "Paper": 1, "Scissors": 1}
}


def get_winner(move1, move2):
    """
    Determines the result of a single round.
    Returns:
        1  if move1 wins,
        -1 if move2 wins,
        0  if it's a draw.
    """
    if move1 == move2:
        return 0
    elif BEATS[move1] == move2:
        return 1
    else:
        return -1


def choose_next_move(previous_move, transition_matrix):
    """
    Player 1 selects their next move based on the transition matrix.
    Uses weighted random sampling according to transition probabilities.
    """
    next_moves = list(transition_matrix[previous_move].keys())
    probabilities = list(transition_matrix[previous_move].values())
    return random.choices(next_moves, weights=probabilities, k=1)[0]


def update_counts(counts, previous_move, current_move):
    """
    Player 2 updates the observed transition counts after each round.
    This is the Bayesian update step: incrementing the count for the
    observed (previous_move → current_move) transition.
    """
    counts[previous_move][current_move] += 1


def estimate_probabilities(counts, previous_move):
    """
    Converts raw transition counts into a probability distribution
    by normalizing over all possible next moves.
    This corresponds to Bayesian inference with a uniform prior.
    """
    total = sum(counts[previous_move].values())
    probabilities = {}

    for move in MOVES:
        probabilities[move] = counts[previous_move][move] / total

    return probabilities


def predict_next_move(counts, previous_move):
    """
    Player 2 predicts Player 1's most likely next move
    by selecting the move with the highest estimated probability.
    """
    probabilities = estimate_probabilities(counts, previous_move)
    predicted_move = max(probabilities, key=probabilities.get)
    return predicted_move


def choose_counter_move(predicted_move):
    """
    Player 2 selects the move that beats the predicted move.
    Implements the counter-move strategy.
    """
    for move in MOVES:
        if BEATS[move] == predicted_move:
            return move


def print_learned_transition_matrix(counts):
    """
    Prints Player 2's learned transition matrix after the simulation.
    Shows the estimated probability of each move given the previous move,
    computed from the accumulated observation counts.
    """
    print("Learned Transition Matrix:")

    for previous_move in MOVES:
        total = sum(counts[previous_move].values())
        print(f"\nAfter {previous_move}:")

        for next_move in MOVES:
            probability = counts[previous_move][next_move] / total
            print(f"  P({next_move} | {previous_move}) = {probability:.3f}")


# --- Simulation Setup ---

num_rounds = 1000

player1_score = 0
player2_score = 0

# Tracks the score difference (Player 2 - Player 1) after each round
accumulated_score = []

# Tracks cumulative wins for each player separately
player1_scores = []
player2_scores = []

# --- Round 0: Both players choose randomly (no prior information yet) ---

player1_move = random.choice(MOVES)
player2_move = random.choice(MOVES)

result = get_winner(player1_move, player2_move)

if result == 1:
    player1_score += 1
elif result == -1:
    player2_score += 1

# Record initial scores
accumulated_score.append(player2_score - player1_score)
player1_scores.append(player1_score)
player2_scores.append(player2_score)

# Store Player 1's first move to use as the starting state for the Markov chain
previous_player1_move = player1_move

# --- Main Simulation Loop (Rounds 1 to 999) ---

for round_number in range(1, num_rounds):

    # Player 2 predicts Player 1's next move using learned transition probabilities
    predicted_move = predict_next_move(learning_counts, previous_player1_move)

    # Player 2 chooses the move that beats the predicted move
    player2_move = choose_counter_move(predicted_move)

    # Player 1 selects next move according to the Markov transition matrix
    player1_move = choose_next_move(previous_player1_move, transition_matrix)

    # Player 2 observes Player 1's actual move and updates transition counts (Bayesian update)
    update_counts(learning_counts, previous_player1_move, player1_move)

    # Determine the winner of this round
    result = get_winner(player1_move, player2_move)

    if result == 1:
        player1_score += 1
    elif result == -1:
        player2_score += 1

    # Record scores after this round
    accumulated_score.append(player2_score - player1_score)
    player1_scores.append(player1_score)
    player2_scores.append(player2_score)

    # Update the previous move for next round's prediction
    previous_player1_move = player1_move


# --- Plotting Results ---

plt.figure(figsize=(12, 5))

# Left plot: cumulative win counts for both players
plt.subplot(1, 2, 1)
plt.plot(player1_scores, label="Player 1")
plt.plot(player2_scores, label="Player 2")
plt.xlabel("Round")
plt.ylabel("Cumulative Score")
plt.title("Cumulative Scores of Both Players")
plt.legend()
plt.grid(True)

# Right plot: score difference over time (positive = Player 2 is ahead)
plt.subplot(1, 2, 2)
plt.plot(accumulated_score)
plt.xlabel("Round")
plt.ylabel("Score Difference (Player 2 - Player 1)")
plt.title("Accumulated Score Difference Over Time")
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Final Results ---

print("Final Score:")
print(f"Player 1: {player1_score}")
print(f"Player 2: {player2_score}")

# Print the transition matrix Player 2 learned through Bayesian updates
print_learned_transition_matrix(learning_counts)