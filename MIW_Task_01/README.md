# Rock–Paper–Scissors with Bayesian Learning

**Evaluation:** Full score (10/10) ✅

This project implements a simulation of the Rock–Paper–Scissors game where an adaptive agent learns and predicts an opponent’s behavior over time.

## Overview

- Player 1 follows a **probabilistic strategy** modeled as a Markov chain
- Player 2 learns transition probabilities using **Bayesian updating**
- The game is simulated over **1000 rounds**
- Performance is tracked and visualized

## How It Works

### Player 1 (Strategy)

- Uses a **transition matrix**
- Next move depends on the previous move
- Represents a **first-order Markov process**

### Player 2 (Learning Agent)

- Observes Player 1’s moves
- Updates transition counts after each round
- Estimates probabilities using normalization (Bayesian inference)
- Predicts the next move
- Chooses the optimal counter-move

## Features

- Markov-based opponent behavior
- Online Bayesian learning
- Adaptive strategy optimization
- Score tracking over time
- Visualization using matplotlib

## Output

- Final scores of both players
- Accumulated score difference
- Learned transition matrix
- Plots of performance over time

## Technologies

- Python
- matplotlib

## How to Run

```bash
python MIW_project_1.py
