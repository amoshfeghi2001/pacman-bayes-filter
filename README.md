# Pacman Bayes Filter — Intelligent Ghost Tracking

**INFO8006 — Introduction to Artificial Intelligence | Project 1**

A Python-based artificial intelligence project that uses **Bayesian Filtering** to track invisible ghosts in Pacman and **Breadth-First Search (BFS)** to navigate the maze and catch them efficiently.

## Project Overview

In this version of Pacman, ghosts are invisible, and their exact positions are unknown. Pacman receives noisy distance measurements from a sensor instead of directly observing the ghosts.

The objective is to build an intelligent agent that can estimate ghost positions, update its beliefs as new information becomes available, and decide where to move.

This project combines **probabilistic reasoning, state estimation, and pathfinding**.

## Key Features

- **Bayesian Filtering:** Estimates ghost locations using movement predictions and sensor observations.
- **Transition Model:** Calculates ghost movement probabilities based on their behavior and Pacman's location.
- **Observation Model:** Handles uncertainty in Manhattan distance measurements.
- **Belief Updates:** Combines predicted positions with sensor evidence.
- **BFS Pathfinding:** Finds shortest paths through the maze.
- **Ghost Tracking:** Supports multiple invisible ghosts and different movement behaviors.

## How It Works

### 1. Transition Model

The `transition_matrix()` function predicts where a ghost may move next.

The model considers available directions, walls, Pacman's position, and the ghost's movement behavior.

Three ghost behaviors are supported:

| Ghost Type | Behavior |
|---|---|
| Fearless | Does not prefer moving away from Pacman |
| Afraid | Prefers moving away from Pacman |
| Terrified | Has a stronger preference for moving away |

### 2. Observation Model

The `observation_matrix()` function calculates the probability of receiving a noisy distance measurement for each possible ghost position.

The sensor uses Manhattan distance with noise drawn from a binomial distribution:

\[
e=d_{\text{Manhattan}}+z-2
\]

where \(z\sim\operatorname{Binomial}(4,0.5)\).

### 3. Bayesian Belief Update

The `update()` function combines the transition and observation models.

It performs two main operations:

1. **Prediction:** Estimate possible new ghost locations from the previous belief state.
2. **Correction:** Update those probabilities using the latest sensor measurement.

The resulting belief distribution is normalized so that its probabilities sum to one.

### 4. Pacman Movement Strategy

The `PacmanAgent` uses Breadth-First Search (BFS) to explore reachable positions and determine shortest-path distances.

The controller combines the belief distributions of uneaten ghosts and scores candidate destinations using:

\[
\text{Target Score}=\frac{\text{Combined Ghost Belief}}{\text{Path Distance}+1}
\]

Pacman follows the first step of the shortest path toward the selected destination. The strategy is recalculated as new observations arrive.

## Technologies

- Python 3.10+
- NumPy
- Bayesian Filtering
- Breadth-First Search (BFS)
- Manhattan Distance
- Git and GitHub
- uv for Python environment management

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/amoshfeghi2001/pacman-bayes-filter.git
cd pacman-bayes-filter
```

### 2. Install dependencies

Install [uv](https://docs.astral.sh/uv/) if it is not already available.

Then run:

```bash
uv sync
```

## Running the Project

### Basic Simulation

Run Pacman against one afraid ghost:

```bash
uv run python run.py --ghost afraid --nghosts 1 --layout large_filter --seed 42
```

### Multiple Ghosts

Run a game with three terrified ghosts:

```bash
uv run python run.py --ghost terrified --nghosts 3 --layout large_filter --seed 42
```

### Maze with Walls

```bash
uv run python run.py --ghost afraid --nghosts 3 --layout large_filter_walls --seed 42
```

### Make Ghosts Visible

For debugging and observing ghost behavior:

```bash
uv run python run.py --ghost afraid --nghosts 3 --layout large_filter --visible
```

### Run Without Graphics

```bash
uv run python run.py --ghost fearless --nghosts 3 --layout large_filter --nographics
```

## Project Structure

```text
pacman-bayes-filter/
├── bayesfilter.py          # Bayesian filter and Pacman agent
├── run.py                  # Game execution script
├── pacman_module/          # Pacman environment
│   ├── ghostAgents.py      # Ghost movement policies
│   ├── game.py             # Game utilities
│   └── layouts/            # Maze configurations
├── test_observation.py     # Observation model tests
├── test_update.py          # Belief update tests
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file
└── README.md               # Project documentation
```

## Implementation Details

The main implementation is located in **`bayesfilter.py`**, which contains:

**BeliefStateAgent**
- `transition_matrix()`
- `observation_matrix()`
- `update()`

**PacmanAgent**
- `_get_action()`

The project uses probabilistic inference to estimate hidden ghost locations and a heuristic pathfinding controller to pursue them.

## Educational Objectives

This project demonstrates how an intelligent agent can:

- Make decisions with incomplete information.
- Represent uncertainty using probability distributions.
- Combine predictions with noisy observations.
- Track moving targets in a dynamic environment.
- Use graph search for navigation and planning.

## Academic Context

Developed as part of **INFO8006 — Introduction to Artificial Intelligence, Project 1**.

The project builds upon the Pacman environment provided for the course.

**GitHub:** [amoshfeghi2001](https://github.com/amoshfeghi2001)
