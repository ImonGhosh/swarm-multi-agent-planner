"""Ant Colony Optimization over the Tree-of-Thought branches (the frozen v2 method).

Branch probability  p_ij = (tau_ij + eps)^alpha / sum_k (tau_ik + eps)^alpha   (no heuristic eta)
Pheromone update    tau  <- (1 - rho) * tau + sum of deposits
Deposit             a trace deposits Q = 1 on each of its edges iff the verifier scores its answer 1.0
"""

DEFAULT_ALPHA = 1.0
DEFAULT_EPSILON = 0.001
DEFAULT_RHO = 0.1
DEFAULT_INITIAL_PHEROMONE = 1.0
