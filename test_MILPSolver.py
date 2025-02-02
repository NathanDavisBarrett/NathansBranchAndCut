from MILP import MILP

from MILPSolver import BranchOnBestBound_MostFractional as Solver
from LinearSolver import SimplexSolver

from LP_Transformer import RepetitiveEqualityTransformer

import numpy as np
from scipy.sparse import csr_matrix

from scipy.optimize import milp as scipy_milp
from scipy.optimize import LinearConstraint as scipy_constr

import time

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
    
def GenerateRandomMILP(numVar,numConstr,lb=-10,ub=10,fracInt=0.5):
    c = np.random.uniform(lb,ub,numVar)

    A = np.random.uniform(lb,ub,(numConstr,numVar))

    xFeas = np.random.uniform(lb,ub,numVar)

    b = A @ xFeas
    
    numInt = int(np.floor(numVar*fracInt))
    integerVars = np.random.choice(numVar,numInt,replace=False)

    milp = MILP(c,integerVars,csr_matrix(A),b,lb=np.ones(numVar)*lb,ub=np.ones(numVar)*ub)
    return milp
    
def SolveMILPWithScipy(milp:MILP):
    constraints = scipy_constr(milp.A_eq,milp.b_eq,milp.b_eq)
    
    integrality = np.zeros(milp.numVar)
    integrality[milp.integerVars] = 1
    
    result = scipy_milp(milp.c,integrality=integrality,constraints=constraints,bounds=(0,None))
    
def SolveMILPWithMySolver(milp:MILP):
    
    logFile = f"test_milp_{time.time()}.log"
    if os.path.exists(logFile):
        os.remove(logFile)

    # linearSolver = SimplexSolver(logLevel=logging.DEBUG, logFile=logFile,logSparseFormat=False)
    linearSolver = SimplexSolver()
    solver = Solver(linearSolver,logLevel=logging.DEBUG,logFile=logFile)

    result= solver.Solve(milp)
    
    return result
    
    
def assertCorrect(milp):
    scipyAnswer = SolveMILPWithScipy(milp)
    myAnswer = SolveMILPWithMySolver(milp)


    #Assert that both answers are feasible
    bPred = milp.A_eq @ scipyAnswer
    assert np.allclose(bPred,milp.b_eq)

    bPred = milp.A_eq @ myAnswer
    assert np.allclose(bPred,milp.b_eq)

    assert np.all(scipyAnswer >= -1e-6)
    assert np.all(myAnswer >= -1e-6)
    
    scipyRounded = np.copy(scipyAnswer)
    scipyRounded[milp.integerVars] = np.round(scipyRounded[milp.integerVars])
    
    myRounded = np.copy(myAnswer)
    myRounded[milp.integerVars] = np.round(myRounded[milp.integerVars])
    
    assert np.allclose(scipyAnswer,scipyRounded)
    assert np.allclose(myAnswer,myRounded)

    #Assert the optimal objective function values are the same
    scipyObj = milp.c @ scipyAnswer
    myObj = milp.c @ myAnswer

    assert np.allclose([scipyObj,],[myObj,])
    
def test_smallRandom():
    milp = GenerateRandomMILP(10,2)
    assertCorrect(milp)

if __name__ == "__main__":
    test_small()
    test_small3d()
    test_smallRandom()