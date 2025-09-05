import numpy as np
from abc import ABC, abstractmethod
import time
import logging

from LP import LP

from TerminationCondition import TerminationCondition


class LinearSolverResult:
    """
    A data class to store the results of linear program optimization.

    This class encapsulates all the important information returned by a linear
    solver, including the optimal solution, basis information, objective value,
    and solver performance metrics.

    Attributes:
        solution (np.array, optional): The optimal solution vector x.
        basis (np.array, optional): Indices of the basic variables.
        obj (float, optional): The optimal objective function value.
        terminationCondition (TerminationCondition, optional): Why the solver terminated.
        numItr (int, optional): Number of iterations performed.
        solveTime (float, optional): Total time spent solving in seconds.
    """

    def __init__(
        self,
        solution=None,
        basis=None,
        obj=None,
        terminationCondition=None,
        numItr=None,
        solveTime=None,
    ):
        """
        Initialize a LinearSolverResult instance.

        Args:
            solution (np.array, optional): The optimal solution vector. Defaults to None.
            basis (np.array, optional): Indices of basic variables. Defaults to None.
            obj (float, optional): Optimal objective value. Defaults to None.
            terminationCondition (TerminationCondition, optional): Termination reason. Defaults to None.
            numItr (int, optional): Number of iterations. Defaults to None.
            solveTime (float, optional): Solve time in seconds. Defaults to None.
        """
        self.solution = solution
        self.basis = basis
        self.obj = obj
        self.terminationCondition = terminationCondition
        self.numItr = numItr
        self.solveTime = solveTime

    def __repr__(self):
        """
        Return a formatted string representation of the solver results.

        Returns:
            str: A human-readable summary of the optimization results including
                 objective value, termination condition, iterations, and solve time.
        """
        term = str(self.terminationCondition).replace("TerminationCondition.", "")
        return f"~~~ SOLVER RESULT~~~\nOptimal Objective Function Value: {self.obj:.5e}\nTermination Condition: {term}\nNumber of Iterations: {self.numItr}\nSolve Time: {self.solveTime:.2f} seconds"


class LinearSolver(ABC):
    """
    An abstract base class for all linear optimization solvers.

    This class provides a common interface and basic functionality for different
    types of linear programming solvers. It includes logging capabilities and
    defines the abstract method that all concrete solvers must implement.

    Attributes:
        logger (logging.Logger): Logger instance for solver output.
        logSparseFormat (bool): Whether to log sparse matrices in sparse format.
    """

    def __init__(
        self,
        logFile=None,
        logLevel=logging.WARNING,
        logger: logging.Logger = None,
        logSparseFormat: bool = True,
    ):
        """
        Initialize the linear solver with logging configuration.

        Args:
            logFile (str, optional): Path to log file. If None, no file logging. Defaults to None.
            logLevel (int, optional): Logging level (e.g., logging.INFO). Defaults to logging.WARNING.
            logger (logging.Logger, optional): Custom logger instance. If None, creates new one. Defaults to None.
            logSparseFormat (bool, optional): Whether to log sparse matrices in sparse format. Defaults to True.
        """
        if logger is None:
            self.logger = logging.getLogger("LINEAR_SOLVER")
            self.logger.setLevel(logLevel)

            if logFile is not None:
                file_handler = logging.FileHandler(logFile)
                file_handler.setLevel(
                    logLevel
                )  # Set the desired log level for the file

                # Create a logging format
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(formatter)

                # Add the file handler to the logger
                self.logger.addHandler(file_handler)
        else:
            self.logger = logger

        self.logSparseFormat = logSparseFormat

        np.set_printoptions(threshold=np.inf)

    @abstractmethod
    def Solve(self, lp: LP) -> LinearSolverResult:
        """
        Solve the given linear program.

        This is an abstract method that must be implemented by all concrete
        linear solver subclasses.

        Args:
            lp (LP): The linear program to solve.

        Returns:
            LinearSolverResult: The solution result containing optimal solution,
                               objective value, and solver information.
        """
        pass


class SimplexSolver(LinearSolver):
    """
    A concrete implementation of the Simplex algorithm for solving linear programs.

    This solver implements the two-phase simplex method for solving linear programs.
    It requires LPs to be in standard form (equality constraints, non-negative variables).
    Use an LP_Transformer to convert general form LPs to standard form before solving.

    The solver supports:
    - Two-phase method for finding initial feasible solutions
    - Customizable pivot rules for variable selection
    - Comprehensive logging and debugging capabilities
    - Sparse matrix operations for efficiency

    Args:
        pivotRule (callable, optional): Function for selecting pivot variable.
            Takes reduced costs vector, returns variable index. Defaults to Bland's rule.
        logFile (str, optional): Path for log file output. Defaults to None.
        logLevel (int, optional): Logging verbosity level. Defaults to logging.WARNING.
        logSparseFormat (bool, optional): Whether to log sparse matrices in sparse format. Defaults to True.
    """

    def __init__(
        self,
        pivotRule=None,
        logFile=None,
        logLevel=logging.WARNING,
        logSparseFormat: bool = True,
    ):
        """
        Initialize the SimplexSolver with specified parameters.

        Args:
            pivotRule (callable, optional): Custom pivot rule function. If None, uses Bland's rule.
            logFile (str, optional): Path to log file. Defaults to None.
            logLevel (int, optional): Logging level. Defaults to logging.WARNING.
            logSparseFormat (bool, optional): Log sparse matrices in sparse format. Defaults to True.
        """
        super().__init__(
            logFile=logFile, logLevel=logLevel, logSparseFormat=logSparseFormat
        )

        if pivotRule is not None:
            self.pivotRule = pivotRule
        else:
            self.pivotRule = lambda cBar: (cBar < -1e-9).argmax(axis=0)

    def FormulateAuxiliaryLP(self, lp: LP) -> LP:
        """
        Generate the auxiliary (Phase I) linear program for finding initial feasible solution.

        The auxiliary LP augments the original constraints with auxiliary variables to
        ensure initial feasibility. The objective becomes minimizing the sum of auxiliary
        variables, which equals zero if and only if the original problem is feasible.

        For the original LP:
            min c^T x
            s.t. A x = b
                 x >= 0

        The auxiliary LP is:
            min sum(u_pos) + sum(u_neg)
            s.t. A x + u_pos - u_neg = b
                 x >= 0, u_pos >= 0, u_neg >= 0

        where u_pos and u_neg are auxiliary variables for positive and negative constraint violations.

        Args:
            lp (LP): The original linear program in standard form.

        Returns:
            LP: The auxiliary linear program with auxiliary variables added.

        Note:
            The auxiliary LP has an obvious initial feasible solution and if its optimal
            value is zero, the original LP is feasible.
        """
        numConstr, numVar = lp.A_eq.shape
        newA = np.hstack([lp.A_eq, np.identity(numConstr), -np.identity(numConstr)])
        newc = np.hstack([np.zeros(numVar), np.ones(2 * numConstr)])

        auxLP = LP(newc, newA, lp.b_eq, lb=np.zeros(len(newc)))
        return auxLP

    def DetermineInitialFeasibleBasis(self, lp: LP, itrLimit, timeLimit):
        """
        Find an initial feasible basis for the linear program using the two-phase method.

        Implements Phase I of the simplex method by solving an auxiliary LP to find
        an initial feasible basic solution for the original problem. If the auxiliary
        problem has optimal value zero, the original problem is feasible.

        Args:
            lp (LP): The linear program in standard form.
            itrLimit (int): Maximum number of iterations allowed.
            timeLimit (float, optional): Maximum time limit in seconds.

        Returns:
            tuple: A tuple containing:
                - basis (np.array or None): Indices of basic variables for initial feasible solution,
                  or None if no feasible solution found within limits.
                - numItr (int): Number of iterations used.
                - terminationCondition (TerminationCondition): Reason for termination.

        Raises:
            Exception: If the auxiliary problem is infeasible or unbounded (indicating
                      a serious error in problem formulation).
        """
        self.logger.info("Determining Initial Feasible Basis...")
        targetBasisSize = lp.A_eq.shape[0]
        auxLP = self.FormulateAuxiliaryLP(lp)
        self.logger.debug(
            "Auxiliary LP:\n%s", auxLP.ToString(sparseFormat=self.logSparseFormat)
        )
        B = np.empty(
            len(auxLP.b_eq), dtype=int
        )  # The initial basis is up[i] if b[i] >= 0 and un[i] if b[i] < 0.
        positive_b = auxLP.b_eq > 0
        negative_b = ~positive_b
        positive_b = np.where(positive_b)[0]
        negative_b = np.where(negative_b)[0]
        B[: len(positive_b)] = (
            positive_b + lp.A_eq.shape[1]
        )  # Select the "up" auxiliary variables for these constraints.
        B[len(positive_b) :] = (
            negative_b + lp.A_eq.shape[1] + lp.A_eq.shape[0]
        )  # Select the "un" auxiliary variables for these constraints.

        result = self.Solve(
            auxLP, B, checkInitialSolution=True, itrLimit=itrLimit, timeLimit=timeLimit
        )

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
            self.logger.warning(
                "Auxiliary problem reached time limit before finding a feasible solution."
            )
            return None, result.numItr, result.terminationCondition

        # If the objective is non-zero, the problem original problem is infeasible.
        if not np.allclose(
            [
                result.obj,
            ],
            [
                0,
            ],
        ):
            self.logger.info("The problem was proven to be infeasible.")
            return None, result.numItr, result.terminationCondition

        # Now determine what the basis of the original LP is.
        originalSolution = result.solution[: lp.A_eq.shape[1]]
        self.logger.debug("Original Solution: %s", originalSolution)

        zeroIndices = np.isclose(originalSolution, np.zeros(len(originalSolution)))
        nonZeroIndices = ~zeroIndices

        zeroIndices = np.where(zeroIndices)[0]
        nonZeroIndices = np.where(nonZeroIndices)[0]

        # Sometimes the initial solution can contain a basic, zero-valued variable. We'll add in enough of those variables to fill the basis here.
        numZerosToSelect = targetBasisSize - len(nonZeroIndices)
        basis = np.hstack([nonZeroIndices, zeroIndices[:numZerosToSelect]])
        self.logger.debug("Initial Feasible Basis: %s", basis)
        return basis, result.numItr + 1, result.terminationCondition

    def AssertStandardForm(self, lp: LP, checkFullRowRank=True):
        """
        Verify that the linear program is in standard form required by the simplex solver.

        Standard form requires:
        - Only equality constraints (no inequalities)
        - No upper bounds on variables
        - Lower bounds should be zero (non-negativity)
        - Optionally, constraint matrix should have full row rank

        Args:
            lp (LP): The linear program to validate.
            checkFullRowRank (bool, optional): Whether to verify that the constraint
                matrix has full row rank. Defaults to True.

        Raises:
            AssertionError: If the LP is not in proper standard form.

        Note:
            This method ensures the LP meets the structural requirements for
            the simplex algorithm implementation.
        """
        lp.AssertValidFormatting()

        numVar = lp.A_eq.shape[1]
        assert lp.A_leq.shape[0] == 0
        assert np.sum(np.isnan(lp.ub)) == numVar
        assert np.allclose(lp.lb, np.zeros(numVar))

        if checkFullRowRank:
            # If this is the case, either there are redundant constraints, or the problem is infeasible. Both situations can be handled by a presolve step. https://en.wikipedia.org/wiki/Revised_simplex_method
            assert (
                lp.A_eq.shape[0] <= numVar
            ), "This LP has more constraints than variables. This indicates the presolve has failed to either remove redundant constraints or has failed to detect infeasibility. In either case, there must be a bug."

    def Solve(
        self,
        lp: LP,
        B: np.array = None,
        checkInitialSolution: bool = True,
        itrLimit=2147483647,
        timeLimit=None,
        computeDualResult=False,
    ) -> LinearSolverResult:
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

        self.logger.info("Beginning Solver Execution...")
        self.logger.debug(
            "Solving the following LP:\n%s",
            lp.ToString(sparseFormat=self.logSparseFormat),
        )

        tic = time.time()
        startItr = 0
        terminationCondition = TerminationCondition.OPTIMAL
        if B is None:
            B, startItr, terminationCondition = self.DetermineInitialFeasibleBasis(
                lp, itrLimit, timeLimit
            )

        if B is None:
            term = (
                TerminationCondition.INFEASIBLE
                if terminationCondition == TerminationCondition.OPTIMAL
                else terminationCondition
            )
            return LinearSolverResult(
                obj=np.infty,
                terminationCondition=term,
                numItr=startItr,
                solveTime=(time.time() - tic),
            )

        if len(B) != lp.A_eq.shape[0]:
            message = "This LP requires a basis size of %s but a basis of size %s was provided."
            self.logger.error(message, lp.A_eq.shape[0], len(B))
            raise Exception(message % (lp.A_eq.shape[0], len(B)))

        self.logger.debug("Basis: %s", B)

        A = lp.A_eq
        A_T = A.transpose()
        c = lp.c
        b = lp.b_eq

        numVar = A.shape[1]

        A_B = A[:, B]
        cB = lp.c[B]

        xB = np.linalg.solve(A_B, b)
        x = np.zeros(numVar)
        x[B] = xB

        self.logger.debug("Initial Solution: %s", x)

        if timeLimit is not None and (time.time() - tic) > timeLimit:
            self.logger.info("Time Limit Exceeded.")
            return LinearSolverResult(
                solution=x,
                obj=c @ x,
                terminationCondition=TerminationCondition.TIME_LIMIT,
                numItr=startItr,
                solveTime=(time.time() - tic),
            )

        if checkInitialSolution:
            self.logger.debug("Checking feasibility of the provided basis...")
            b_replicate = A @ x
            result = np.allclose(b, b_replicate)
            assert result, "Something has gone wrong. Ax != b"

            assert np.sum(xB <= -1e-8) == 0, "The initial basis is infeasible."
            self.logger.debug("Provided basis is feasible.")

        unbounded = False
        itrTermination = False
        timeTermination = False
        for simplexIteration in range(startItr, itrLimit + 1):
            self.logger.debug(
                "~~~ STARTING SIMPLEX ITERATION #%s ~~~", simplexIteration
            )
            self.logger.debug("A_B:\n%s", A_B)
            if simplexIteration >= itrLimit:
                self.logger.info("Iteration Limit Reached.")
                itrTermination = True
                break
            if timeLimit is not None and (time.time() - tic) > timeLimit:
                self.logger.info("Time Limit Reached.")
                timeTermination = True
                break

            # Step 1: Compute reduced costs
            # Step 1-a: Solve for "p"
            p = np.linalg.solve(A_B.T, cB)

            # Step 1-b: Solve for reduced costs
            cBar = c - A_T @ p
            self.logger.debug("cBar: %s", cBar)

            # Step 2: Check optimality conditions
            negativeReducedCosts = cBar < -1e-9
            if np.sum(negativeReducedCosts) == 0:
                self.logger.info("Optimal Solution Found.")
                break

            # Step 3: Select j
            j = self.pivotRule(cBar)
            self.logger.debug("j: %s", j)
            if j in B:
                raise Exception("Attempting to pivot a basis index into the basis!")

            # Step 4: Select direction
            jCol = (-A[:, j]).flatten()
            dB = np.linalg.solve(A_B, jCol)
            self.logger.debug("dB: %s", dB)

            # Step 5: Check if the problem is unbounded
            boundedIndices = np.where(dB < -1e-9)[0]
            self.logger.debug("boundedIndices: %s", boundedIndices)
            if len(boundedIndices) == 0:
                self.logger.info("Problem was proven to be unbounded.")
                unbounded = True
                break

            # Step 6: Compute step length
            overallBoundedIndices = B[boundedIndices]
            candidates = -x[overallBoundedIndices] / dB[boundedIndices]
            self.logger.debug("Step Length Candidates: %s", candidates)
            minIndex = np.argmin(candidates)
            theta = candidates[minIndex]
            minIndex = boundedIndices[minIndex]
            self.logger.debug("Step Length: %s", theta)
            self.logger.debug(
                "Restricting Variable Index (w.r.t. to basis): %s", minIndex
            )

            # Step 7: Update x
            dispacement = np.zeros(numVar)
            dispacement[B] = theta * dB
            dispacement[j] = theta
            self.logger.debug("Displacement: %s", dispacement)

            x += dispacement
            self.logger.debug("new x: %s", x)

            # Step 8: Update Basis
            self.logger.debug("Removing variable %s from the basis.", B[minIndex])
            self.logger.debug("Adding variable %s to the basis.", j)
            B[minIndex] = j
            self.logger.debug("New Basis: %s", B)
            cB[minIndex] = c[j]
            A_B[:, minIndex] = A[:, j]

        terminationCondition = TerminationCondition.OPTIMAL
        if itrTermination:
            terminationCondition = TerminationCondition.ITR_LIMIT
        if timeTermination:
            terminationCondition = TerminationCondition.TIME_LIMIT

        self.logger.info("Optimization Complete.")

        if unbounded:
            terminationCondition = TerminationCondition.UNBOUNDED
            result = LinearSolverResult(
                obj=-np.infty,
                terminationCondition=terminationCondition,
                numItr=simplexIteration,
                solveTime=(time.time() - tic),
            )
        else:
            result = LinearSolverResult(
                solution=x,
                basis=B,
                obj=c @ x,
                terminationCondition=terminationCondition,
                numItr=simplexIteration,
                solveTime=(time.time() - tic),
            )

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
                y = np.linalg.solve(-A_B.T, cB)
                dualResult = LinearSolverResult(
                    solution=y,
                    obj=-result.obj,
                    terminationCondition=TerminationCondition.OPTIMAL,
                    numItr=result.numItr,
                    solveTime=result.solveTime,
                )
            elif terminationCondition == TerminationCondition.INFEASIBLE:
                dualResult = LinearSolverResult(
                    solution=None,
                    obj=-np.infty,
                    terminationCondition=TerminationCondition.UNBOUNDED,
                    numItr=result.numItr,
                    solveTime=result.solveTime,
                )
            elif terminationCondition == TerminationCondition.UNBOUNDED:
                dualResult = LinearSolverResult(
                    solution=None,
                    obj=np.infty,
                    terminationCondition=TerminationCondition.INFEASIBLE,
                    numItr=result.numItr,
                    solveTime=result.solveTime,
                )
            else:
                self.logger.warning(
                    'Only Optimal, Infeasible, of Unbounded solutions can directly produce dual solutions, not "%s"',
                    terminationCondition,
                )
                dualResult = None

            self.logger.debug("Dual Solution: %s", dualResult.solution)

            return result, dualResult

        else:
            return result
