from abc import ABC, abstractmethod

from LinearSolver import SimplexSolver
from TerminationCondition import TerminationCondition
from MILP import MILP
from LP import LP
from LinkedList import LinkedList, SortedLinkedList, LinkedListNode

import logging
import numpy as np
import time


class MILPSolverResult:
    """
    Container for mixed-integer linear program solver results.

    Stores the results of solving a MILP including the optimal integer solution,
    objective values, performance metrics, and termination information.

    Attributes:
        solution (np.array, optional): The optimal integer solution.
        obj (float, optional): The optimal integer objective value.
        relaxObj (float, optional): The optimal LP relaxation objective value.
        terminationCondition (TerminationCondition, optional): Why the solver terminated.
        numNodesExplored (int, optional): Number of branch-and-bound nodes explored.
        solveTime (float, optional): Total solve time in seconds.
        mipGap (float): The MIP optimality gap between integer and relaxed solutions.
    """

    def __init__(
        self,
        solution=None,
        obj=None,
        relaxObj=None,
        terminationCondition=None,
        numNodesExplored=None,
        solveTime=None,
    ):
        """
        Initialize a MILP solver result.

        Args:
            solution (np.array, optional): Optimal solution vector. Defaults to None.
            obj (float, optional): Optimal integer objective value. Defaults to None.
            relaxObj (float, optional): LP relaxation objective value. Defaults to None.
            terminationCondition (TerminationCondition, optional): Termination reason. Defaults to None.
            numNodesExplored (int, optional): Number of B&B nodes explored. Defaults to None.
            solveTime (float, optional): Total solve time in seconds. Defaults to None.
        """
        self.solution = solution
        self.obj = obj
        self.relaxObj = relaxObj
        self.terminationCondition = terminationCondition
        self.numNodesExplored = numNodesExplored
        self.solveTime = solveTime
        self.mipGap = np.abs((self.obj - self.relaxObj) / self.obj)

    def __repr__(self):
        """
        Return a formatted string representation of the MILP solver results.

        Returns:
            str: Human-readable summary including objective values, termination
                 condition, iterations, solve time, and MIP gap.
        """
        term = str(self.terminationCondition).replace("TerminationCondition.", "")
        return f"~~~ SOLVER RESULT~~~\nOptimal Integer Objective Function Value: {self.obj:.5e}\nOptimal Relaxed Objective Function Value: {self.relaxObj:.5e}\nTermination Condition: {term}\nNumber of Iterations: {self.numItr}\nSolve Time: {self.solveTime:.2f} seconds\nMIP Gap: {self.gap:.5e}"


class UnboundedError(Exception):
    """Exception raised when the LP relaxation is unbounded."""

    pass


class NodeSolutionError(Exception):
    """Exception raised when there's an error solving a branch-and-bound node."""

    pass


class MILPSolver(ABC):
    """
    Abstract base class for Mixed-Integer Linear Program solvers.

    Provides the framework for implementing branch-and-bound and other MILP
    solution algorithms. Handles logging, LP relaxation solving, and common
    MILP solver functionality.

    Attributes:
        linearSolver (SimplexSolver): The LP solver used for relaxations.
        logger (logging.Logger): Logger for solver output and debugging.
    """

    def __init__(
        self,
        linearSolver: SimplexSolver,
        logFile=None,
        logLevel=logging.WARNING,
        logger: logging.Logger = None,
    ):
        """
        Initialize the MILP solver.

        Args:
            linearSolver (SimplexSolver): LP solver for solving relaxations.
            logFile (str, optional): Path to log file. Defaults to None.
            logLevel (int, optional): Logging level. Defaults to logging.WARNING.
            logger (logging.Logger, optional): Custom logger. If None, creates new one. Defaults to None.
        """
        if logger is None:
            self.logger = logging.getLogger("MILP_SOLVER")
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

        self.linearSolver = linearSolver

    @abstractmethod
    def Solve(self, milp: MILP) -> MILPSolverResult:
        """
        A function that takes in a milp and returns a MILPSolverResult object
        """
        pass


class BranchAndCut(MILPSolver):
    """
    To begin, we will relax all the integer constraints in the MILP (thus transforming the problem into an LP). We then will seek to find the solution to this "relaxed" LP. We will then add constraints back into the LP that re-introduce the integer constraints. But we don't want to have to re-solve the problem every time we add a new constraint. Thus, we will seek to solve the Dual of the original relaxed LP.

    Thus, we want the initially provided lp to be in a form whose dual problem is in standard form. Thus, we want the initially provided problem to be in the following form:

    ###### PROBLEM 1 ######
    min c^T x
    s.t. A x <= b

    The dual of this problem is

    ###### PROBLEM 1-D ######
    min b^T y
    s.t. -A^T y == c
         y >= 0

    Which is in standard form (and can thus be solved directly by the simplex solver). I'll refer to this primal form whose dual is in standard form as the "dual standard form".

    Adding a set of constraints (A' x <= b') changes the primal problem to the following:

    ###### PROBLEM 2 ######
         ┌─ ─┐^T ┌─ ─┐
    min  │ c │   │ x │
         └─ ─┘   └─ ─┘

    s.t. ┌─    ─┐┌─ ─┐    ┌─    ─┐
         │┌─  ─┐││   │    │┌─  ─┐│
         │  A   ││   │    │  b   │
         │└─  ─┘││ x │ <= │└─  ─┘│
         │┌─  ─┐││   │    │┌─  ─┐│
         │  A'  ││   │    │  b'  │
         │└─  ─┘││   │    │└─  ─┘│
         └─    ─┘└─ ─┘    └─    ─┘

    Thus the dual of this new problem is as follows:
    ###### PROBLEM 2-D ######
         ┌─    ─┐^T┌─    ─┐
         │┌─  ─┐│  │┌─  ─┐│
         │  b   │  │  y   │
    min  │└─  ─┘│  │└─  ─┘│
         │┌─  ─┐│  │┌─  ─┐│
         │  b'  │  │  y'  │
         │└─  ─┘│  │└─  ─┘│
         └─    ─┘  └─    ─┘

    s.t. ┌─                ─┐┌─    ─┐    ┌─ ─┐
         │┌─     ─┐┌─     ─┐││┌─  ─┐│    │   │
         │  -A^T     -A'^T  ││  y   │ == │ c │
         │└─     ─┘└─     ─┘││└─  ─┘│    │   │
         └─                ─┘│┌─  ─┐│    └─ ─┘
                             │  y'  │
                             │└─  ─┘│
                             └─    ─┘
         [y, y'] >= 0

    Given a solution to Problem 1-D we can compute the associated solution to Problem 1 using complimentary slackness. In english that means that any constraint in Problem 1 that corresponds with a non-zero-valued value in the optimal solution of Problem 1-D is a binding constraint. The collection of these binding constraints (A* and b*) forms a linear equation (A* x == b*) that can be solved to determine the solution to Problem 1.

    From this solution, we can add constraints which further guide the LP-relaxed problem to the true MILP solution (e.g. a bounding constraint, a cut, etc.). These added constraints can be directly added to Problem 2-D maintaining that the optimal solution fo Problem 1-D is still feasible for Problem 2-D (preventing us from having the re-solve the problem from scratch at every iteration).

    The key then becomes: How do we known which constraints to add and when?

    KEEP GOING HERE!
    """

    class Tree:
        """
        A class to house information about the b & b tree used in the solution of this problem.
        """

        def __init__(
            self,
            milp: MILP,
            lpSolver: SimplexSolver,
            logFile=None,
            logLevel=logging.WARNING,
            logger: logging.Logger = None,
        ):
            if logger is None:
                self.logger = logging.getLogger("MILP_SOLVER")
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

            self.headNode = self.HeadNode(milp, self.logger)
            self.c = milp.c
            self.lpSolver = lpSolver

            self.integerVars = milp.integerVars

            # A list of the active nodes to be considered SORTED by which one has the best bound.
            #   "Active" means a node that has been solved and who might have child nodes of interest.
            self.ActiveNodes = SortedLinkedList([], key=lambda x: x.primalSolution.obj)

            self.incumbentNode = None
            self.incumbentObj = None

            self.maxDepth = 0
            self.numTotalNodes = 1

            self.SolveNode(self.headNode)

        class Node(ABC):
            def __init__(
                self,
                c: np.ndarray,
                integerVars: np.ndarray,
                depth: int,
                coordinate: str,
                logger: logging.Logger,
            ):
                self.c = c
                self.integerVars = integerVars
                self.depth = depth
                self.coordinate = coordinate

                self.logger = logger

                self.solved = False
                self.dualSolution = None
                self.primalSolution = None
                self.isIntegral = None

                self.childNodes = set([])

            @abstractmethod
            def Assemble_A_b(self, leaf=True):
                """
                A function that create the A_eq and b_eq components for the "Problem 2-D" formulation of this node.
                """
                pass

            def Solve(self, lpSolver: SimplexSolver):
                """
                A function to solve the LP associated with this node.
                """
                A, b = self.Assemble_A_b(leaf=True)
                duaLp = LP(c=b, A_eq=-A.transpose(), b_eq=self.c, lb=np.zeros(len(b)))
                if (
                    hasattr(self, "parentNode")
                    and self.parentNode.dualSolution.terminationCondition
                    == TerminationCondition.OPTIMAL
                ):
                    B = np.copy(self.parentNode.dualSolution.basis)
                else:
                    B = None
                result = lpSolver.Solve(
                    duaLp, B=B, computeDualResult=True
                )  # Recall the dual of the dual is the primal. Thus from the node's perspective the original primal problem is the dual of the node's problem and vice versa.
                try:
                    self.dualSolution, self.primalSolution = result
                except TypeError:
                    self.dualSolution = result
                    self.primalSolution = None
                    self.logger.debug(
                        'Node "%s" solved with infeasible primal termination condition',
                        self.coordinate,
                    )

                self.solved = True

                self.DetermineIntegrality()

                if self.primalSolution is not None:
                    self.logger.debug(
                        'Node "%s" solved with primal termination condition of %s and an integrality status of %s',
                        self.coordinate,
                        self.primalSolution.terminationCondition,
                        self.isIntegral,
                    )
                    self.logger.debug(
                        'Node "%s" produced the following primal solution: %s',
                        self.coordinate,
                        self.primalSolution.solution,
                    )

            def DetermineIntegrality(self):
                """
                A function to determine and assign the value of self.isIntegral
                """
                if self.primalSolution is None:
                    self.isIntegral = False
                elif (
                    self.primalSolution.terminationCondition
                    != TerminationCondition.OPTIMAL
                ):
                    self.isIntegral = False
                else:
                    supposedIntegerVars = self.primalSolution.solution[self.integerVars]
                    integerValues = np.rint(supposedIntegerVars)
                    error = np.abs(supposedIntegerVars - integerValues)
                    self.isIntegral = np.all(error <= 1e-6)

            def __hash__(self):
                return id(self)

        class StandardNode(Node):
            """
            A class to house a node of the B & B tree.
            """

            def __init__(
                self,
                c: np.array,
                integerVars: np.array,
                parentNode,
                addedConstraints_A: np.array,
                addedConstraints_b: np.array,
                depth: int,
                coordinate: str,
                logger,
            ):
                super().__init__(c, integerVars, depth, coordinate, logger)
                self.addedConstraints_A = addedConstraints_A
                self.addedConstraints_b = addedConstraints_b

                self.parentNode = parentNode

            def Assemble_A_b(self, leaf=True):
                """
                A function to assemble the A_eq and b_eq components of "Problem 2-D" for this node that is represented by this node.

                Each node represents a series of added constraints imposed beyond that node's parent LP. Thus by walking up the tree to the head node, a series of constraints that were put in place to reach the current node can be assembled. These constraints will be of the form A' x <= b'.

                We desire to assemble A' and b'. This will be done by vertically stacking rows together. But to make the stacking of these rows as easy as possible. We'll collect each grouping of added constraint in a linked list (for rapid linking of different results)
                """
                # Step 1, collect my added constraints.
                aList = LinkedList(reverseIteration=True)
                bList = LinkedList(reverseIteration=True)

                aList.append(self.addedConstraints_A)
                bList.append(self.addedConstraints_b)

                # Step 2, collect added constraints from parent node.
                result = self.parentNode.Assemble_A_b(leaf=False)
                aList.extend(result[0])
                bList.extend(result[1])

                # Step 3, return the results
                if leaf:
                    A = np.vstack(list(aList))
                    b = np.hstack(
                        tuple(bList)
                    )  # TODO: hstack currently requires a __getitem__ method. A pull request was submitted to fix this problem. Thus the tuple conversion might not be required in the future.
                    return A, b
                else:
                    return aList, bList

        class HeadNode(Node):
            """
            The head node of a B & B tree.
            """

            def __init__(self, problem: MILP, logger: logging.Logger):
                super().__init__(problem.c, problem.integerVars, 0, "head", logger)
                self.baseLP = problem

            def Assemble_A_b(self, leaf=True):
                # Step 1, collect my added constraints.
                aList = LinkedList(reverseIteration=True)
                bList = LinkedList(reverseIteration=True)

                aList.append(self.baseLP.A_leq)
                bList.append(self.baseLP.b_leq)

                # Step 2, return the results
                if leaf:
                    A = np.vstack(list(aList))
                    b = np.hstack(
                        tuple(bList)
                    )  # TODO: hstack currently requires a __getitem__ method. A pull request was submitted to fix this problem. Thus the tuple conversion might not be required in the future.
                    return A, b
                else:
                    return aList, bList

        def SolveNode(self, node: Node):
            """
            A function to solve a given node and handle whether or not it should now be considered an active node or not.

            This function also returns a boolean indicating whether or not the solution of this node has lead to a new incumbent solution or not.
            """
            node.Solve(self.lpSolver)
            if (
                node.primalSolution is not None
                and node.primalSolution.terminationCondition
                == TerminationCondition.OPTIMAL
            ):
                obj = node.primalSolution.obj
                if node.isIntegral:
                    if self.incumbentObj is None or obj < self.incumbentObj:
                        self.incumbentObj = obj
                        self.incumbentNode = node
                        return True
                    else:
                        # This is a sub-optimal integer solution. Do not consider this node as active.
                        return False
                else:
                    if self.incumbentObj is not None and obj >= self.incumbentObj:
                        # This relaxed LP is still worse than the best integer solution. Do not consider this node as active
                        return False
                    else:
                        # This node could still yield integral child nodes that out-perform the incumbent solution. Label this node as "Active"
                        self.ActiveNodes.insert(node)
                        return False
            elif (
                node.primalSolution is None
                or node.primalSolution.terminationCondition
                == TerminationCondition.INFEASIBLE
            ):
                # There is no use in pursuing any child nodes of this node since they'll all be infeasible. Do not consider this node as active.
                return False
            elif (
                node.primalSolution.terminationCondition
                == TerminationCondition.UNBOUNDED
            ):
                # An un-bounded child problem indicates an unbounded master problem.
                raise UnboundedError()
            else:
                # An un-determined sub-problem will cause problems for the MILP solver overall.
                raise NodeSolutionError(
                    f"Node was able to reach an acceptable terminal condition. It's termination condition was {node.primalSolution.terminationCondition}"
                )

        @property
        def bestBoundNode(self):
            if self.ActiveNodes.headNode is None:
                return None
            return self.ActiveNodes.headNode.value

        @property
        def bestBoundObj(self):
            if self.bestBoundNode is None or self.bestBoundNode.primalSolution is None:
                return np.infty
            return self.bestBoundNode.primalSolution.obj

        @property
        def bestRGap(self):
            if self.incumbentObj is None or self.bestBoundObj is None:
                return np.infty
            return np.abs((self.bestBoundObj - self.incumbentObj) / self.incumbentObj)

        @property
        def bestAGap(self):
            if self.incumbentObj is None or self.bestBoundObj is None:
                return np.infty
            return np.abs((self.bestBoundObj - self.incumbentObj))

        def _draw_graph_with_shapes_and_colors(self, G, pos):
            """
            This function is copied from ChatGPT.
            """
            import networkx as nx

            shapes = set((node[1]["shape"] for node in G.nodes(data=True)))

            for shape in shapes:
                nodes_with_shape = [
                    node for node in G.nodes(data=True) if node[1]["shape"] == shape
                ]
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=[node[0] for node in nodes_with_shape],
                    node_color=[node[1]["color"] for node in nodes_with_shape],
                    node_shape=shape,
                )

            nx.draw_networkx_edges(G, pos)
            nx.draw_networkx_labels(G, pos)

        def Plot(self):
            import matplotlib.pyplot as plt
            import networkx as nx
            from matplotlib.lines import Line2D
            from networkx.drawing.nx_pydot import graphviz_layout

            def add_edges(graph: nx.Graph, node):
                if (
                    self.incumbentNode is not None
                    and node.coordinate == self.incumbentNode.coordinate
                ):
                    shape = "s"
                    color = "gold"
                elif (
                    self.bestBoundNode is not None
                    and node.coordinate == self.bestBoundNode.coordinate
                ):
                    shape = "o"
                    color = "lawngreen"
                elif node.isIntegral:
                    shape = "s"
                    color = "deepskyblue"
                else:
                    shape = "o"
                    color = "deepskyblue"

                graph.add_node(node.coordinate, shape=shape, color=color)
                for child in node.childNodes:
                    graph.add_edge(node.coordinate, child.coordinate)
                    add_edges(graph, child)

            graph = nx.DiGraph()
            add_edges(graph, self.headNode)

            pos = graphviz_layout(graph, prog="dot")
            self._draw_graph_with_shapes_and_colors(graph, pos)

            incumbentPatch = Line2D(
                [0],
                [0],
                color="gold",
                marker="s",
                linestyle="None",
                markersize=10,
                label="Incumbent Solution",
            )
            bestBoundPatch = Line2D(
                [0],
                [0],
                color="lawngreen",
                marker="o",
                linestyle="None",
                markersize=10,
                label="Best Bound",
            )
            integerPatch = Line2D(
                [0],
                [0],
                color="deepskyblue",
                marker="s",
                linestyle="None",
                markersize=10,
                label="Integer Solution",
            )
            fractionalPatch = Line2D(
                [0],
                [0],
                color="deepskyblue",
                marker="o",
                linestyle="None",
                markersize=10,
                label="Fractional Solution",
            )

            plt.legend(
                handles=[incumbentPatch, bestBoundPatch, integerPatch, fractionalPatch]
            )

            plt.show()

    def Branch(self, tree: Tree, llNode: LinkedListNode, variable: int):
        """
        The code to handle the branching of a given node upon a given variable.

        This function also returns a boolean indicating whether or not the solution of either of the resulting child nodes lead to an improved incumbent solution or not.

        Arguments
        ---------
        tree: Tree
            The tree within which this node resides.
        llNode: LinkedListNode
            The LinkedListNode within the tree.ActiveNodes LinkedList containing the Tree.Node that you'd like to branch on.
        variable int
            The index of the primal variable you'd like to branch on.

        Returns
        -------
        newIncumbent: bool
            A boolean indicating whether or not either of the child nodes produced from this branch lead to a new incumbent solution.
        """
        node = llNode.value

        varVal = node.primalSolution.solution[variable]
        minVal = int(np.floor(varVal))
        maxVal = minVal + 1

        self.logger.debug(
            'Branching at node "%s" on variable %s with value %s',
            node.coordinate,
            variable,
            varVal,
        )

        addedA = np.zeros((1, len(node.c)))
        addedA[0, variable] = 1

        newDepth = node.depth + 1

        minNode = self.Tree.StandardNode(
            c=node.c,
            integerVars=node.integerVars,
            parentNode=node,
            addedConstraints_A=addedA,
            addedConstraints_b=np.array(
                [
                    minVal,
                ]
            ),
            depth=newDepth,
            coordinate=f"N{tree.numTotalNodes}",  # node.coordinate + f"-{variable}_l_{minVal}",
            logger=tree.logger,
        )
        maxNode = self.Tree.StandardNode(
            c=node.c,
            integerVars=node.integerVars,
            parentNode=node,
            addedConstraints_A=-addedA,
            addedConstraints_b=np.array(
                [
                    -maxVal,
                ]
            ),
            depth=newDepth,
            coordinate=f"N{tree.numTotalNodes+1}",  # node.coordinate + f"-{variable}_g_{maxVal}",
            logger=tree.logger,
        )

        node.childNodes.add(minNode)
        node.childNodes.add(maxNode)

        if newDepth > tree.maxDepth:
            tree.maxDepth = newDepth
        tree.numTotalNodes += 2

        # Since this will no longer be a leaf node, we will not consider it as active any more.
        tree.ActiveNodes.remove(llNode)

        minIncumb = tree.SolveNode(minNode)
        maxIncumb = tree.SolveNode(maxNode)
        return minIncumb or maxIncumb

    def AddCuts(
        self, tree: Tree, llNode: LinkedListNode, cut_A: np.array, cut_b: np.array
    ):
        """
        The code the handle the addition of cuts to a given node.

        Arguments
        ---------
        tree: Tree
            The tree within which this node resides.
        llNode: LinkedListNode
            The LinkedListNode within the tree.ActiveNodes LinkedList containing the Tree.Node that you'd like to add cuts to.
        cut_A: csr_matrix
            A matrix containing the coefficients for the added cuts in cut_A x <= cut_b form.
        cut_b: np.ndarray
            A numpy array containing the constants for the added cuts in cut_A x <= cut_b form.

        Returns
        -------
        newIncumb: bool
            A boolean indicating whether or not the child node resulting from these cuts resulted in a new incumbent solution.
        """
        node = llNode.value

        self.logger.debug(
            'Adding the following cut to node "%s": %s \n %s',
            node.coordinate,
            cut_A,
            cut_b,
        )

        newDepth = node.depth + 1
        newNode = self.Tree.StandardNode(
            c=node.c,
            integerVars=node.integerVars,
            parentNode=node,
            addedConstraints_A=cut_A,
            addedConstraints_b=cut_b,
            depth=newDepth,
            coordinate=f"N{tree.numTotalNodes}",  # node.coordinate + "-ct",
            logger=tree.logger,
        )

        node.childNodes.add(newNode)

        if newDepth > tree.maxDepth:
            tree.maxDepth = newDepth
        tree.numTotalNodes += 1

        # Don't consider the parent node active any more:
        tree.ActiveNodes.remove(llNode)

        return tree.SolveNode(newNode)

    def TestOptimal(self, tree: Tree):
        """
        For a B & C - based MILP solver, the optimum is determined when there are no more active nodes to consider.
        """
        numActive = len(tree.ActiveNodes)
        self.logger.debug("There are %s active nodes.", numActive)
        return numActive == 0

    def PruneViaOptimality(self, tree: Tree):
        """
        Some nodes are self-pruned when they are solved (e.g. the node is infeasible or produces a lp-relaxed obj. that is inferior to the incumbent solution). But some nodes might be considered active until the incumbent solution is improved upon. In such a case, a sweep should be done to prune and nodes that might become inactive due to the new incumbent solution. This function executes this sweep.
        """
        nodei = tree.ActiveNodes.headNode
        while nodei is not None:
            if nodei.value.primalSolution.obj >= tree.incumbentObj:
                self.logger.debug(
                    'Pruning node "%s" with relaxed obj. of %s due to improved incumbent solution (%s).',
                    nodei.value.coordinate,
                    nodei.value.primalSolution.obj,
                    tree.incumbentObj,
                )
                tree.ActiveNodes.remove(nodei)
            nodei = nodei.nextNode

    @abstractmethod
    def DetermineAction(self, tree):
        """
        A function that takes in the current solution tree and determines which action should be taken next (e.g. whether or not to branch or cut, which node and/or variable to branch or cut on).

        Arguments
        ---------
        tree: BranchAndCut.Tree
            The solution tree to consider

        Returns
        -------
        func: python function
            A python function to indicate the action to take. The first argument should be the tree object passed in. Any additional arguments or key-word arguments should be provided from the other output of the DetermineAction function.
        args: tuple
            Other arguments to pass to the "func"
        kwargs: dict
            Other key-word arguments to pass to the "func"
        """
        pass

    def Solve(
        self,
        milp: MILP,
        aGap: float = 0.0,
        rGap: float = 0.0,
        timeLimit: float = np.infty,
        returnTree: bool = False,
    ) -> MILPSolverResult:
        self.logger.info("Starting MILP solver run.")
        self.logger.debug("Constructing solver tree.")
        tree = self.Tree(milp, self.linearSolver, logger=self.logger)

        term = TerminationCondition.OPTIMAL
        tic = time.time()
        toc = time.time()

        while not self.TestOptimal(tree):
            toc = time.time()
            if tree.bestAGap < aGap:
                self.logger.info("Optimization terminated due to absolute gap limit.")
                term = TerminationCondition.ABS_GAP_LIMIT
                break
            if tree.bestRGap < rGap:
                self.logger.info("Optimization terminated due to relative gap limit.")
                term = TerminationCondition.REL_GAP_LIMIT
                break
            if toc - tic > timeLimit:
                self.logger.info("Optimization terminated due to time limit.")
                term = TerminationCondition.TIME_LIMIT
                break

            self.logger.debug("Determining action...")
            action, args, kwargs = self.DetermineAction(tree)
            try:
                newIncumbent = action(tree, *args, **kwargs)
            except UnboundedError:
                self.logger.info("Problem was proven to be unbounded.")
                term = TerminationCondition.UNBOUNDED
                break
            if newIncumbent:
                self.logger.debug(
                    "New incumbent solution found. Attempting to prune via optimality."
                )
                self.PruneViaOptimality(tree)

        if (
            (tree.incumbentObj is None)
            and (term is not TerminationCondition.UNBOUNDED)
            and (len(tree.ActiveNodes) == 0)
        ):
            self.logger.debug(
                "The problem terminated naturally with no feasible solution."
            )
            self.logger.info("The problem was proven to be infeasible.")
            term = TerminationCondition.INFEASIBLE
            tree.incumbentObj = np.infty

        self.logger.info("Optimization complete.")
        if term == TerminationCondition.OPTIMAL:
            self.logger.info("Optimal solution found.")

        if tree.incumbentNode is not None:
            result = MILPSolverResult(
                solution=tree.incumbentNode.primalSolution.solution,
                obj=tree.incumbentObj,
                relaxObj=tree.bestBoundObj,
                terminationCondition=term,
                numNodesExplored=tree.numTotalNodes,
                solveTime=toc - tic,
            )
        else:
            result = MILPSolverResult(
                solution=None,
                obj=tree.incumbentObj,
                relaxObj=tree.bestBoundObj,
                terminationCondition=term,
                numNodesExplored=tree.numTotalNodes,
                solveTime=toc - tic,
            )

        if returnTree:
            return result, tree
        else:
            return result


class BranchOnBestBound_MostFractional(BranchAndCut):
    def DetermineAction(self, tree: BranchAndCut.Tree):
        """
        This function always selects to branch from the node with the best (most optimistic) node and on the integral variable that currently has the most fractional value.
        """
        bestboundLLNode = tree.ActiveNodes.headNode
        bestBoundNode = bestboundLLNode.value
        solution = bestBoundNode.primalSolution.solution

        targetIntegers = solution[tree.integerVars]
        distances = np.abs(targetIntegers - np.round(targetIntegers))
        furthestIndex = np.argmax(distances)

        branchVarIndex = tree.integerVars[furthestIndex]

        return self.Branch, (bestboundLLNode, branchVarIndex), {}


# TODO: Strong branching
# TODO: Cut generation
# TODO: Heuristic solution
