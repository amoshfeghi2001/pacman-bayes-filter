
from bayesfilter import BeliefStateAgent
from pacman_module.game import Grid

walls = Grid(7, 7)
agent = BeliefStateAgent("afraid")

O = agent.observation_matrix(
    walls=walls,
    evidence=3,
    position=(1, 1),
)

print("Distance 3:", O[4, 1])
print("Distance 2:", O[3, 1])
print("Distance 0:", O[1, 1])

assert O.shape == (7, 7)
assert O[4, 1] == 0.375
assert O[3, 1] == 0.25
assert O[1, 1] == 0.0

print("All observation tests passed!")
