from MILP import MILP

from MILPSolver import BranchOnBestBound_MostFractional as Solver
from LinearSolver import SimplexSolver

from LP_Transformer import RepetitiveEqualityTransformer

import numpy as np
from scipy.sparse import csr_matrix

import logging, os

def test_small():
    A = np.array([
        [0.1,-1],
        [0.1,1],
        [1,0]
    ])
    b = np.array([-2.15,4.85,10.2])

    milp = MILP(
        c=np.array([-1,-1],dtype=float),
        integerVars=np.array([0,1],dtype=int),
        A_leq=csr_matrix(A),
        b_leq=b
    )

    logFile = "test_small.log"
    if os.path.exists(logFile):
        os.remove(logFile)

    # linearSolver = SimplexSolver(logLevel=logging.DEBUG, logFile=logFile,logSparseFormat=False)
    linearSolver = SimplexSolver()
    solver = Solver(linearSolver,logLevel=logging.DEBUG,logFile=logFile)

    result = solver.Solve(milp,timeLimit=10)

    expected = np.array([8,4])

    assert np.allclose(expected,result.solution)

def test_small3d():
    A = np.array([
        [0.1,-1,0],
        [0.1,1,-1],
        [1,0,10],
        [0.3,-0.8,2],
        [0,0,1]
    ])
    b = np.array([-1.95,3.05,9.5,-0.5,0])

    milp = MILP(
        c=np.array([-1,-1,-1],dtype=float),
        integerVars=np.array([0,1,2],dtype=int),
        A_leq=csr_matrix(A),
        b_leq=b
    )

    logFile = "test_small3d.log"
    if os.path.exists(logFile):
        os.remove(logFile)

    # linearSolver = SimplexSolver(logLevel=logging.DEBUG, logFile=logFile,logSparseFormat=False)
    linearSolver = SimplexSolver()
    solver = Solver(linearSolver,logLevel=logging.DEBUG,logFile=logFile)

    result= solver.Solve(milp,timeLimit=10)


    expected = np.array([0,3,0])

    assert np.allclose(expected,result.solution)

