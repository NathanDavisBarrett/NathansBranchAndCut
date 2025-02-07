from enum import Enum

class TerminationCondition(Enum):
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
    ITR_LIMIT = 4
    TIME_LIMIT = 5
    ABS_GAP_LIMIT = 6
    REL_GAP_LIMIT = 7