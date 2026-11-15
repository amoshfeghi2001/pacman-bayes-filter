import numpy as np

from pacman_module.game import Agent, Directions, manhattanDistance

from collections import deque


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """

        width = walls.width
        height = walls.height

        T = np.zeros((width, height, width, height))

        fear = {
            "fearless": 0,
            "afraid": 1,
            "terrified": 3,
        }[self.ghost]

        moves = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]

        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    continue

                old_distance = manhattanDistance((x, y), position)

                candidates = []
                weights = []

                for dx, dy in moves:
                    nx = x + dx
                    ny = y + dy

                    if not (0 <= nx < width and 0 <= ny < height):
                        continue

                    if walls[nx][ny]:
                        continue

                    new_distance = manhattanDistance((nx, ny), position)

                    if new_distance >= old_distance:
                        weight = 2**fear
                    else:
                        weight = 1

                    candidates.append((nx, ny))
                    weights.append(weight)

                total = sum(weights)

                if total == 0:
                    T[x, y, x, y] = 1.0
                else:
                    for (nx, ny), weight in zip(candidates, weights):
                        T[x, y, nx, ny] = weight / total

        return T

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """

        width = walls.width
        height = walls.height

        O = np.zeros((width, height))

        probabilities = [1, 4, 6, 4, 1]

        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    continue

                true_distance = manhattanDistance(position, (x, y))

                noise = evidence - true_distance

                if -2 <= noise <= 2:
                    O[x, y] = probabilities[noise + 2] / 16

        return O

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """

        T = self.transition_matrix(walls, position)
        O = self.observation_matrix(walls, evidence, position)

        predicted = np.zeros_like(belief, dtype=float)

        for x in range(walls.width):
            for y in range(walls.height):
                predicted += belief[x, y] * T[x, y]

        posterior = predicted * O

        total = posterior.sum()

        if total > 0:
            return posterior / total

        predicted_total = predicted.sum()

        if predicted_total > 0:
            return predicted / predicted_total

        return np.array(belief, dtype=float, copy=True)

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

    def _get_action(self, walls, beliefs, eaten, position):

        if all(eaten):
            return Directions.STOP

        combined = np.zeros((walls.width, walls.height))

        for i in range(len(beliefs)):
            if not eaten[i]:
                combined += beliefs[i]

        moves = [
            (0, 1, Directions.NORTH),
            (0, -1, Directions.SOUTH),
            (1, 0, Directions.EAST),
            (-1, 0, Directions.WEST),
        ]

        queue = deque([(position, None, 0)])
        visited = {position}

        best_score = -1
        best_action = Directions.STOP

        while queue:
            (x, y), first_action, distance = queue.popleft()

            if distance > 0:
                score = combined[x, y] / (distance + 1)

                if score > best_score:
                    best_score = score
                    best_action = first_action

            for dx, dy, action in moves:
                nx = x + dx
                ny = y + dy
                next_position = (nx, ny)

                if not (0 <= nx < walls.width):
                    continue

                if not (0 <= ny < walls.height):
                    continue

                if walls[nx][ny]:
                    continue

                if next_position in visited:
                    continue

                visited.add(next_position)

                next_action = first_action

                if next_action is None:
                    next_action = action

                queue.append((next_position, next_action, distance + 1))

        return best_action

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
