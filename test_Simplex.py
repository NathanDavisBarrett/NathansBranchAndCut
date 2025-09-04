from LinearSolver import SimplexSolver
from TerminationCondition import TerminationCondition
from LP import LP
from LP_Transformer import BasicTransformer

import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linprog as scipy_lp


def GenerateRandomLP(numVar, numConstr, lb=-10, ub=10):
    c = np.random.uniform(lb, ub, numVar)

    A = np.random.uniform(lb, ub, (numConstr, numVar))

    xFeas = np.random.uniform(lb, ub, numVar)

    b = A @ xFeas

    lp = LP(c, csr_matrix(A), b, lb=np.ones(numVar) * lb, ub=np.ones(numVar) * ub)
    trans = BasicTransformer(lp)

    transLP = trans.Transform()
    return transLP

<<<<<<< Updated upstream
def SolveLPWithScipy(lp:LP):
    numConstr,numVar = lp.A_eq.shape
    
    result = scipy_lp(lp.c,A_eq=lp.A_eq,b_eq=lp.b_eq,bounds=(0,None))

    return result.x
=======

def SolveLPWithCvxpy(lp: LP):
    numConstr, numVar = lp.A_eq.shape

    x = cp.Variable(numVar)
    prob = cp.Problem(cp.Minimize(lp.c @ x), [lp.A_eq @ x == lp.b_eq, x >= 0])
    prob.solve()
    return x.value
>>>>>>> Stashed changes


def SolveWithMySolver(lp, B=None):
    solver = SimplexSolver(logLevel=10)
    result = solver.Solve(lp, B=B)
    return result.solution


def assertCorrect(lp):
    cvxpyAnswer = SolveLPWithScipy(lp)
    myAnswer = SolveWithMySolver(lp)

    # Assert that both answers are feasible
    bPred = lp.A_eq @ cvxpyAnswer
    assert np.allclose(bPred, lp.b_eq)

    bPred = lp.A_eq @ myAnswer
    assert np.allclose(bPred, lp.b_eq)

    assert np.all(cvxpyAnswer >= -1e-6)
    assert np.all(myAnswer >= -1e-6)

    # Assert the optimal objective function values are the same
    cvxpyObj = lp.c @ cvxpyAnswer
    myObj = lp.c @ myAnswer

    assert np.allclose(
        [
            cvxpyObj,
        ],
        [
            myObj,
        ],
    )


def test_0():
    """
    Simple LP: maximize 2*x1 + x2
    subject to:
      x1 + x2 == 4
      2x1  == 6
      x1 >= 0, x2 >= 0
    """
    c = np.array([2, 1])
    A_eq = np.array([[1, 1], [2, 0]])
    b_eq = np.array([4, 6])

    lp = LP(c=c, A_eq=csr_matrix(A_eq), b_eq=b_eq, lb=np.zeros(2))
    assertCorrect(lp)


def test_1():
    A = np.array(
        [
            # x1,x2,x3,s1,s2,s3,s4,s5,s6,s7
            [-1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    b = np.array([0, 0, 0, 2, 2, 2, 4]).transpose()

    c = np.array([1, 1, -1, 0, 0, 0, 0, 0, 0, 0]).transpose()

    lp = LP(c=c, A_eq=csr_matrix(A), b_eq=b, lb=np.zeros(len(c)))
    assertCorrect(lp)


def test_random1():
    np.random.seed(1)
    lp = GenerateRandomLP(2, 2)
    assertCorrect(lp)


def test_random10():
    """
    Test 50 random feasible LPs
    """
    np.random.seed(1)
    for i in range(10):
        numVar = np.random.randint(3, 10)
        numConstr = np.random.randint(2, numVar)
        lp = GenerateRandomLP(numVar, numConstr)
        assertCorrect(lp)


def test_Infeasible():
    lp = LP(
        c=np.array([1, 0]),
        A_leq=csr_matrix(np.array([[-1, 1], [1, -1]])),
        b_leq=np.array([0, -1]),
    )
    trans = BasicTransformer(lp)
    lp = trans.Transform()

    solver = SimplexSolver()
    result = solver.Solve(lp)

    assert result.terminationCondition == TerminationCondition.INFEASIBLE


def test_Unbounded():
    lp = LP(
        c=np.array([1, 0]),
        A_leq=csr_matrix(np.array([[1, -1], [-1, 1]])),
        b_leq=np.array([0, 1]),
    )
    trans = BasicTransformer(lp)
    lp = trans.Transform()

    solver = SimplexSolver()
    result = solver.Solve(lp)

    assert result.terminationCondition == TerminationCondition.UNBOUNDED


def test_ItrLimit():
    np.random.seed(1)
    lp = GenerateRandomLP(1000, 1000)
    solver = SimplexSolver()
    result = solver.Solve(lp, itrLimit=100)
    assert result.terminationCondition == TerminationCondition.ITR_LIMIT


def test_TimeLimit():
    np.random.seed(1)
    lp = GenerateRandomLP(1000, 1000)
    solver = SimplexSolver()
    result = solver.Solve(lp, timeLimit=10)
    assert result.terminationCondition == TerminationCondition.TIME_LIMIT



if __name__ == "__main__":
    test_random1()
    test_random10()
    test_Infeasible()
    test_Unbounded()
    test_ItrLimit()
    test_TimeLimit()