from scipy.sparse import csr_matrix, identity, hstack, vstack
from scipy.sparse.linalg import splu, spsolve_triangular, spsolve


import numpy as np
from abc import ABC,abstractmethod
from warnings import warn
import time
import logging

from LP import LP

from TerminationCondition import TerminationCondition


class LinearSolverResult:
    def __init__(self,solution=None,basis=None,obj=None,terminationCondition=None,numItr=None,solveTime=None):
        self.solution = solution
        self.basis = basis
        self.obj = obj
        self.terminationCondition = terminationCondition
        self.numItr = numItr
        self.solveTime = solveTime

    def __repr__(self):
        term = str(self.terminationCondition).replace("TerminationCondition.","")
        return f"~~~ SOLVER RESULT~~~\nOptimal Objective Function Value: {self.obj:.5e}\nTermination Condition: {term}\nNumber of Iterations: {self.numItr}\nSolve Time: {self.solveTime:.2f} seconds"
        

class LinearSolver(ABC):
    """
    An abstract base class for all linear solvers.
    """
    def __init__(self,logFile=None,logLevel=logging.WARNING,logger:logging.Logger=None,logSparseFormat:bool=True):
        if logger is None:
            self.logger = logging.getLogger("LINEAR_SOLVER")
            self.logger.setLevel(logLevel)

            if logFile is not None:
                file_handler = logging.FileHandler(logFile)
                file_handler.setLevel(logLevel)  # Set the desired log level for the file

                # Create a logging format
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)

                # Add the file handler to the logger
                self.logger.addHandler(file_handler)
        else:
            self.logger = logger

        self.logSparseFormat = logSparseFormat

        np.set_printoptions(threshold=np.inf)

    @abstractmethod
    def Solve(self,lp:LP)-> LinearSolverResult:
        """
        A function that takes in an lp and returns a LinearSolverResult object
        """
        pass


class SimplexSolver(LinearSolver):
    """
    A class to house the simplex solver.

    For this solver, LPs must be transformed to the following form:

        min c^T x
        s.t. A == x
             x >= 0

        There are several ways to do this. Please select an LP_Transformer object to perform this task before passing the lp to this solver.

    Required Arguments
    ------------------
    pivotRule: function (Default = Bland's Rule)
        A function that takes in a vector of reduced costs and returns the index of the variable to pivot on.
    """
    def __init__(self, pivotRule=None,logFile=None, logLevel=logging.WARNING,logSparseFormat:bool=True):
        super().__init__(logFile=logFile,logLevel=logLevel,logSparseFormat=logSparseFormat)

        if pivotRule is not None:
            self.pivotRule = pivotRule
        else:
            self.pivotRule = lambda cBar: (cBar < -1e-9).argmax(axis=0)

    def solve_triangular(self,L,b,*args,**kwargs):
        if L.size > 1:    
            return spsolve_triangular(L,b,*args,**kwargs)
        else:
            #Scipy's sparse solvers can't handle a 1X1 matrix.
            return b / L[0,0]
        
    def lu(self,A):
        if A.size > 1:    
            LU = splu(A)
            L = LU.L
            U = LU.U
            n = LU.perm_r.shape[0]
            Pr = csr_matrix((np.ones(n), (LU.perm_r, np.arange(n))),shape=(n,n))
            Pc = csr_matrix((np.ones(n), (np.arange(n), LU.perm_c)),shape=(n,n))
        else:
            L = A
            U = Pr = Pc = csr_matrix(np.identity(1))
        return L,U,Pr,Pc

    def SolveSystemUsingLUFactorizing(self,L,U,Pr,Pc,b,transposeLU=False):
        if not transposeLU:
            p = Pr @ b
            self.solve_triangular(L,p,lower=True,overwrite_b=True)
            self.solve_triangular(U,p,lower=False,overwrite_b=True)
            return Pc @ p
        else:
            #Remember that (AB)^T = B^T A^T
            p = Pc.transpose() @ b
            self.solve_triangular(U.transpose(),p,lower=True,overwrite_b=True)
            self.solve_triangular(L.transpose(),p,lower=False,overwrite_b=True)
            return Pr.transpose() @ p
        
    def presolve(self,lp:LP) -> LP:
        """
        A function to perform presolve operations on an LP to remove redundant constraints and detect infeasibility.

        Arguments
        ---------
        lp: LP
            The linear program you'd like to perform presolve operations on.

        Returns
        -------
        lp: LP
            The linear program after presolve operations have been performed.
        """
        self.AssertStandardForm(lp,checkFullRowRank=False)
        self.logger.debug("Beginning Presolve Operations...")

        # Find the LU factorization of A_eq
        L,U,Pr,Pc = self.lu(lp.A_eq)

        p = Pr @ lp.b_eq
        self.solve_triangular(L,p,lower=True,overwrite_b=True)

        # Check for infeasibilities
        if np.any(np.isnan(p) | np.isinf(p)):
            self.logger.info("Problem is infeasible.")
            return TerminationCondition.INFEASIBLE
        
        self.solve_triangular(U,p,lower=False,overwrite_b=True)

        #Check for infeasibilities
        if np.any(np.isnan(p) | np.isinf(p)):
            self.logger.info("Problem is infeasible.")
            return TerminationCondition.INFEASIBLE
        
        #Check for redundant constraints
        redundantRows = Find rows of U that are all zero.

        originalRedundantRows = Pr[redundantRows,:].indices #CHECK THIS, ITS FROM CHATGPT

        A_reduced = lp.A_eq[~originalRedundantRows,:]
        b_reduced = lp.b_eq[~originalRedundantRows]

        # Now assemble the new LP
        KEEP GIONG HERE!


        self.AssertStandardForm(lp,checkFullRowRank=True)

    def FormulateAuxiliaryLP(self,lp:LP) -> LP:
        """
        A function to generate the auxiliary LP for a given LP.

        The auxiliary LP of an original LP is the same LP but with all constraints augmented with auxiliary variables to make every constraint feasible.

        The objective of this auxiliary LP is to drive all auxiliary variables to zero.

        In math, given the following LP

        min c^T X
        s.t. A x == b
             x >= 0

        The auxiliary problem is formulated as follows (here "u" are the auxiliary variables added to each constraint.)

        min up + un
        s.t. A x + up - un == b
             x >= 0
             up >= 0
             un >= 0

        The auxiliary problem has a known feasible initial basis: up[i] if b[i] >= 0 and un[i] if b[i] < 0.


        Arguments
        ---------
        lp: LP
            The linear program you'd like to generate the auxiliary LP for.

        Returns
        -------
        auxLP: LP
            The auxiliary LP
        """
        numConstr,numVar = lp.A_eq.shape
        newA = hstack([lp.A_eq,identity(numConstr),-identity(numConstr)]).tocsr()
        newc = np.hstack([np.zeros(numVar),np.ones(2*numConstr)])

        auxLP = LP(newc,newA,lp.b_eq,lb=np.zeros(len(newc)))
        return auxLP
        
    def DetermineInitialFeasibleBasis(self,lp:LP,itrLimit,timeLimit):
        self.logger.info("Determining Initial Feasible Basis...")
        targetBasisSize = lp.A_eq.shape[0]
        auxLP = self.FormulateAuxiliaryLP(lp)
        self.logger.debug(f"Auxiliary LP:\n%s",auxLP.ToString(sparseFormat=self.logSparseFormat))
        B = np.empty(len(auxLP.b_eq),dtype=int) #The initial basis is up[i] if b[i] >= 0 and un[i] if b[i] < 0.
        positive_b = auxLP.b_eq > 0
        negative_b = ~positive_b
        positive_b = np.where(positive_b)[0]
        negative_b = np.where(negative_b)[0]
        B[:len(positive_b)] = positive_b + lp.A_eq.shape[1] #Select the "up" auxiliary variables for these constraints.
        B[len(positive_b):] = negative_b + lp.A_eq.shape[1] + lp.A_eq.shape[0] #Select the "un" auxiliary variables for these constraints.


        result = self.Solve(auxLP,B,checkInitialSolution=True,itrLimit=itrLimit,timeLimit=timeLimit)

        if result.terminationCondition == TerminationCondition.INFEASIBLE:
            message = "A problem occurred... Auxiliary problem is infeasible."
            self.logger.error(message)
            raise Exception(message)
        elif result.terminationCondition == TerminationCondition.UNBOUNDED:
            message = "A problem occurred... Auxiliary problem is unbounded."
            self.logger.error(message)
            raise Exception(message)
        elif result.terminationCondition == TerminationCondition.ITR_LIMIT:
            message = "Auxiliary problem reached iteration limit before finding a feasible solution."
            self.logger.warning(message)
            return None, result.numItr, result.terminationCondition
        elif result.terminationCondition == TerminationCondition.TIME_LIMIT:
            self.logger.warning("Auxiliary problem reached time limit before finding a feasible solution.")
            return None, result.numItr, result.terminationCondition

        #If the objective is non-zero, the problem original problem is infeasible.
        if not np.allclose([result.obj,],[0,]):
            self.logger.info("The problem was proven to be infeasible.")
            return None, result.numItr, result.terminationCondition
        
        #Now determine what the basis of the original LP is.
        originalSolution = result.solution[:lp.A_eq.shape[1]]
        self.logger.debug("Original Solution: %s",originalSolution)

        zeroIndices = np.isclose(originalSolution,np.zeros(len(originalSolution)))
        nonZeroIndices = ~zeroIndices

        zeroIndices = np.where(zeroIndices)[0]
        nonZeroIndices = np.where(nonZeroIndices)[0]

        #Sometimes the initial solution can contain a basic, zero-valued variable. We'll add in enough of those variables to fill the basis here.
        numZerosToSelect = targetBasisSize - len(nonZeroIndices)
        basis = np.hstack([nonZeroIndices,zeroIndices[:numZerosToSelect]])
        self.logger.debug("Initial Feasible Basis: %s",basis)
        return basis, result.numItr + 1, result.terminationCondition
    
    def AssertStandardForm(self,lp:LP,checkFullRowRank=True):
        lp.AssertValidFormatting()

        numVar = lp.A_eq.shape[1]
        assert lp.A_leq.shape[0] == 0
        assert np.sum(np.isnan(lp.ub)) == numVar
        assert np.allclose(lp.lb,np.zeros(numVar))

        if checkFullRowRank:
            # If this is the case, either there are redundant constraints, or the problem is infeasible. Both situations can be handled by a presolve step. https://en.wikipedia.org/wiki/Revised_simplex_method
            assert lp.A_eq.shape[0] <= numVar, "This LP has more constraints than variables. This indicates the presolve has failed to either remove redundant constraints or has failed to detect infeasibility. In either case, there must be a bug."

    def Solve(self,lp:LP,B:np.array=None,checkInitialSolution:bool=True,itrLimit=2147483647,timeLimit=None,computeDualResult=False) -> LinearSolverResult:
        """
        Arguments
        ---------
        lp: LP
            The linear program you're wanting to solve
        B: np.array (optional, Default = None)
            The indices of the basic variables in the initial basic solution. If None is provided, an auxiliary LP will be formulated and solved to determine one.
        checkInitialSolution: bool (optional, Default = None)
            An indication of whether or not to check that the initial basic solution is feasible. If it is not, an auxiliary LP will be formulated and solved to determine one.
        maxItr: int (optional, Default = 2147483647):
            The maximum number of iterations to use in this optimization.
        timeLimit: float (optional, Default = None)
            The maximum number of seconds you'd like the solver to run for. If None is provided, no time limit will be imposed.
        computeDualResult: bool (optional, Default = False)
            An indication of whether or not you'd like to return a LinearSolverResult for both the primal AND dual solutions.
        """
        self.presolve(lp)

        self.logger.info("Beginning Solver Execution...")
        self.logger.debug("Solving the following LP:\n%s",lp.ToString(sparseFormat=self.logSparseFormat))

        tic = time.time()
        startItr = 0
        terminationCondition = TerminationCondition.OPTIMAL
        if B is None:
            B,startItr,terminationCondition = self.DetermineInitialFeasibleBasis(lp,itrLimit,timeLimit)

        if B is None:
            term = TerminationCondition.INFEASIBLE if terminationCondition == TerminationCondition.OPTIMAL else terminationCondition
            return LinearSolverResult(obj=np.infty,terminationCondition=term,numItr=startItr,solveTime=(time.time() - tic))
        
        if len(B) != lp.A_eq.shape[0]:
            message = "This LP requires a basis size of %s but a basis of size %s was provided."
            self.logger.error(message,lp.A_eq.shape[0],len(B))
            raise Exception(message % (lp.A_eq.shape[0],len(B)))
        
        self.logger.debug("Basis: %s",B)
        
        A = lp.A_eq
        A_T = A.transpose()
        c = lp.c
        b = lp.b_eq

        numVar = A.shape[1]
   
        A_B = A[:,B]
        cB = lp.c[B]

        L,U,Pr,Pc = self.lu(A_B)
        
        xB = self.SolveSystemUsingLUFactorizing(L,U,Pr,Pc,b)
        x = np.zeros(numVar)
        x[B] = xB

        self.logger.debug("Initial Solution: %s",x)

        if timeLimit is not None and (time.time() - tic) > timeLimit:
            self.logger.info("Time Limit Exceeded.")
            return LinearSolverResult(solution=x,obj=c@x,terminationCondition=TerminationCondition.TIME_LIMIT,numItr=startItr,solveTime=(time.time() - tic))

        if checkInitialSolution:
            self.logger.debug("Checking feasibility of the provided basis...")
            b_replicate = A @ x
            result = np.allclose(b,b_replicate)
            assert result, "Something has gone wrong. Ax != b"

            assert np.sum(xB <= -1e-8) == 0, "The initial basis is infeasible."
            self.logger.debug("Provided basis is feasible.")

        unbounded = False
        itrTermination = False
        timeTermination = False
        for simplexIteration in range(startItr,itrLimit+1):
            self.logger.debug("~~~ STARTING SIMPLEX ITERATION #%s ~~~",simplexIteration)
            self.logger.debug("A_B:\n%s",A_B.toarray())
            if simplexIteration >= itrLimit:
                self.logger.info("Iteration Limit Reached.")
                itrTermination = True
                break
            if timeLimit is not None and (time.time() - tic) > timeLimit:
                self.logger.info("Time Limit Reached.")
                timeTermination = True
                break

            #Step 1: Compute reduced costs
            #Step 1-a: Solve for "p"
            p = self.SolveSystemUsingLUFactorizing(L,U,Pr,Pc,cB,transposeLU=True)

            #Step 1-b: Solve for reduced costs
            cBar = c - A_T @ p
            self.logger.debug("cBar: %s",cBar)

            #Step 2: Check optimality conditions
            negativeReducedCosts = cBar < -1e-9
            if np.sum(negativeReducedCosts) == 0:
                self.logger.info("Optimal Solution Found.")
                break

            #Step 3: Select j
            j = self.pivotRule(cBar)
            self.logger.debug("j: %s",j)
            if j in B:
                raise Exception("Attempting to pivot a basis index into the basis!")
            
            #Step 4: Select direction
            jCol = (-A[:,j]).toarray().flatten()
            dB = self.SolveSystemUsingLUFactorizing(L,U,Pr,Pc,jCol).flatten()
            self.logger.debug("dB: %s",dB)

            #Step 5: Check if the problem is unbounded
            boundedIndices = np.where(dB < -1e-9)[0]
            self.logger.debug("boundedIndices: %s",boundedIndices)
            if len(boundedIndices) == 0:
                self.logger.info("Problem was proven to be unbounded.")
                unbounded = True
                break
            
            #Step 6: Compute step length
            overallBoundedIndices = B[boundedIndices]
            candidates = -x[overallBoundedIndices] / dB[boundedIndices]
            self.logger.debug("Step Length Candidates: %s",candidates)
            minIndex = np.argmin(candidates)
            theta = candidates[minIndex]
            minIndex = boundedIndices[minIndex]
            self.logger.debug("Step Length: %s",theta)
            self.logger.debug("Restricting Variable Index (w.r.t. to basis): %s",minIndex)

            #Step 7: Update x
            dispacement = np.zeros(numVar)
            dispacement[B] = theta * dB
            dispacement[j] = theta
            self.logger.debug("Displacement: %s",dispacement)

            x += dispacement
            self.logger.debug("new x: %s",x)

            #Step 8: Update Basis
            self.logger.debug("Removing variable %s from the basis.",B[minIndex])
            self.logger.debug("Adding variable %s to the basis.",j)
            B[minIndex] = j
            self.logger.debug("New Basis: %s",B)
            cB[minIndex] = c[j]
            A_B[:,minIndex] = A[:,j]

            L,U,Pr,Pc = self.lu(A_B)

        terminationCondition = TerminationCondition.OPTIMAL
        if itrTermination:
            terminationCondition = TerminationCondition.ITR_LIMIT
        if timeTermination:
            terminationCondition = TerminationCondition.TIME_LIMIT

        self.logger.info("Optimization Complete.")

        if unbounded:
            terminationCondition = TerminationCondition.UNBOUNDED
            result = LinearSolverResult(obj=-np.infty,terminationCondition=terminationCondition,numItr=simplexIteration,solveTime=(time.time() - tic))
        else:
            result = LinearSolverResult(solution=x,basis=B,obj=c @ x, terminationCondition=terminationCondition, numItr=simplexIteration, solveTime=(time.time() - tic))

        if computeDualResult:
            """
            Recall that the dual problem for an LP of the form

            min c^T x
            s.t. A x == b
                x >= 0

            is

            min b^T y
            s.t. -A^T y <= c

            Given an optimal solution to the primal problem, any primal-basic variables will correspond with binding constraints in the dual problem (due to complementarity).

            Thus, "y" can be solved for using the following relationship:

            (-A^T)[B,:] y == c[B]
            """
            if terminationCondition == TerminationCondition.OPTIMAL:
                y = self.SolveSystemUsingLUFactorizing(-L,U,Pr,Pc,cB,transposeLU=True)
                dualResult = LinearSolverResult(
                    solution=y,
                    obj=-result.obj,
                    terminationCondition=TerminationCondition.OPTIMAL,
                    numItr=result.numItr,
                    solveTime=result.solveTime
                )
            elif terminationCondition == TerminationCondition.INFEASIBLE:
                dualResult = LinearSolverResult(
                    solution=None,
                    obj = -np.infty,
                    terminationCondition=TerminationCondition.UNBOUNDED,
                    numItr=result.numItr,
                    solveTime=result.solveTime
                )
            elif terminationCondition == TerminationCondition.UNBOUNDED:
                dualResult = LinearSolverResult(
                    solution=None,
                    obj = np.infty,
                    terminationCondition=TerminationCondition.INFEASIBLE,
                    numItr=result.numItr,
                    solveTime=result.solveTime
                )
            else:
                self.logger.warning("Only Optimal, Infeasible, of Unbounded solutions can directly produce dual solutions, not \"%s\"", terminationCondition)
                dualResult = None

            self.logger.debug("Dual Solution: %s", dualResult.solution)
            
            return result, dualResult

        else:
            return result
    