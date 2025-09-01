from enum import Enum

class TerminationCondition(Enum):
    """
    Enumeration of possible termination conditions for optimization solvers.
    
    This enum defines all the possible reasons why an optimization algorithm
    might terminate, allowing for consistent reporting across different solvers.
    
    Values:
        OPTIMAL: The solver found an optimal solution.
        INFEASIBLE: The problem has no feasible solution.
        UNBOUNDED: The problem is unbounded (objective can be improved indefinitely).
        ITR_LIMIT: Maximum number of iterations reached.
        TIME_LIMIT: Maximum time limit exceeded.
        ABS_GAP_LIMIT: Absolute optimality gap tolerance reached.
        REL_GAP_LIMIT: Relative optimality gap tolerance reached.
    """
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
    ITR_LIMIT = 4
    TIME_LIMIT = 5
    ABS_GAP_LIMIT = 6
    REL_GAP_LIMIT = 7