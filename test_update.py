
import numpy as np

from bayesfilter import BeliefStateAgent
from pacman_module.game import Grid

# Create a 5x5 maze without walls.
walls = Grid(5, 5)

# Create a fearless ghost model.
agent = BeliefStateAgent("fearless")

# Create the old belief matrix.
belief = np.zeros((5, 5))

# We believe the ghost starts at (2, 2).
belief[2, 2] = 1.0

# Pacman is at (0, 0).
# The sensor reports distance 5.
new_belief = agent.update(
    walls=walls,
    belief=belief,
    evidence=5,
    position=(0, 0),
)

# Display the results.
print("Right:", new_belief[3, 2])
print("Left:", new_belief[1, 2])
print("Up:", new_belief[2, 3])
print("Down:", new_belief[2, 1])
print("Total:", new_belief.sum())

# Verify the result.
assert np.isclose(new_belief[3, 2], 6 / 14)
assert np.isclose(new_belief[1, 2], 1 / 14)
assert np.isclose(new_belief[2, 3], 6 / 14)
assert np.isclose(new_belief[2, 1], 1 / 14)
assert np.isclose(new_belief.sum(), 1.0)

print("Update test passed!")
