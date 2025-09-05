from abc import ABC, abstractmethod

from LP import LP

import numpy as np


class LP_Transformer:
    """
    Abstract base class for transforming linear programs to standard form.

    Facilitates the conversion of general form linear programs to the standard form
    required by simplex-based solvers. Standard form eliminates inequality constraints
    and variable bounds, converting them to equality constraints with non-negative variables.

    Input LP form:
        min c^T x
        s.t. A_eq x = b_eq
             A_leq x <= b_leq
             lb <= x <= ub

    Output standard form:
        min c^T x
        s.t. A x = b
             x >= 0

    Attributes:
        originalLP (LP): The original linear program to be transformed.

    Note:
        Subclasses must implement Transform() and UnTransform() methods.
    """

    def __init__(self, lp: LP):
        """
        Initialize the transformer with a linear program.

        Args:
            lp (LP): The linear program to be transformed.
        """
        self.originalLP = lp

    @abstractmethod
    def Transform(self):
        """
        Transform the linear program to standard form.

        This method must be implemented by subclasses to convert the original
        LP to standard form suitable for simplex algorithms.

        Returns:
            LP: The transformed linear program in standard form.
        """
        pass

    @abstractmethod
    def UnTransform(self, x):
        """
        Convert solution from transformed space back to original space.

        Takes a solution to the transformed LP and returns the corresponding
        solution to the original LP.

        Args:
            x (np.array): Solution vector from the transformed LP.

        Returns:
            np.array: Solution vector for the original LP.
        """
        pass

    def IsStandardForm(self, lp: LP):
        """
        Determine whether a linear program is in standard form.

        Standard form requires:
        - No inequality constraints (A_leq x <= b_leq)
        - No upper bounds on variables
        - All lower bounds should be zero (non-negativity)

        Args:
            lp (LP): The linear program to check.

        Returns:
            bool: True if the LP is in standard form, False otherwise.
        """
        numLeq = len(lp.b_leq)
        if numLeq != 0:
            return False

        noUB = np.all(np.isnan(lp.ub))
        if not noUB:
            return False

        allZeroLB = np.allclose(lp.lb, np.zeros(len(lp.lb)))
        if not allZeroLB:
            return False

        return True


class BasicTransformer(LP_Transformer):
    """
    This transformer works by reading the inputted LP as is and accomplishing the transformation by taking advantage of the fact that any inequality

        a x <= b

    Can be transformed into the appropriate form by introducing a slack variable.

        a x + s == b
        s >= 0

    However, since the original "x" variables can be positive or negative, we also need to define positive and negative components for each variable.

        x = xp - xn
        xp >= 0
        xn >= 0

    Thus, any equality constraint given as a x == b should be reformulated as a xp - a xn == b.
    Any inequality constraint given as a x <= b should be reformulated as a xp - a xn + s == b.

    Thus, the final form of the transformed LP A x == b constraint is as follows:
    ┌─                                  ─┐ ┌─         ─┐    ┌─         ─┐
    │ ┌─       ─┐ ┌─       ─┐ ┌─      ─┐ │ │  ┌─    ─┐ │    │ ┌─     ─┐ │
    │     A_eq      - A_eq        0      │ │     xp    │    │    b_eq   │
    │ └─       ─┘ └─       ─┘ └─      ─┘ │ │           │    │ └─     ─┘ │
    | ┌─       ─┐ ┌─       ─┐ ┌─      ─┐ │ │  └─    ─┘ │    │ ┌─     ─┐ │
    │    A_leq      - A_leq              │ │  ┌─    ─┐ │    │   b_leq   │
    │ └─       ─┘ └─       ─┘            │ │     xn    │ == │ └─     ─┘ │
    | ┌─       ─┐ ┌─       ─┐            │ │           │    │ ┌─     ─┐ │
    |     -I'          I'         I      │ │  └─    ─┘ │    │    -lb    │
    │ └─       ─┘ └─       ─┘            │ │  ┌─    ─┐ │    │ └─     ─┘ │
    | ┌─       ─┐ ┌─       ─┐            │ │     s     │    │ ┌─     ─┐ │
    |      I''        -I''               │ │           │    │     ub    │
    │ └─       ─┘ └─       ─┘ └─      ─┘ │ │  └─    ─┘ │    │ └─     ─┘ │
    └─                                  ─┘ └─         ─┘    └─         ─┘

    Where I' is the identity matrix with rows removed that have their "1" value in a column that represents a variable that does not have a lower bound. Likewise, I'' is the identity matrix with rows removed that have their "1" value in a column that represents a variable that does not have an upper bound.
    """

    def Transform(self):
        originalNumVar = self.originalLP.A_eq.shape[1]

        # The first step is to determine the number of slack variables needed.
        #   Since this transformer it aimed at being as simple as possible, no simplifications to the inputted LP will be made.
        #   This the number of slack variables needed is equal to the total number of inequalities.
        numSlack = (
            self.originalLP.A_leq.shape[0]
            + np.count_nonzero(~np.isnan(self.originalLP.lb))
            + np.count_nonzero(~np.isnan(self.originalLP.ub))
        )

        # Next up we need to assemble I' and I''
        varsWithLowerBound = ~np.isnan(self.originalLP.lb)
        varsWithUpperBound = ~np.isnan(self.originalLP.ub)

        varsWithLowerBound = (
            np.argwhere(varsWithLowerBound).flatten()
            if sum(varsWithLowerBound) > 0
            else np.array([], dtype=int)
        )
        varsWithUpperBound = (
            np.argwhere(varsWithUpperBound).flatten()
            if sum(varsWithUpperBound) > 0
            else np.array([], dtype=int)
        )

        nVarLB = len(varsWithLowerBound)
        nVarUB = len(varsWithUpperBound)

        Ip = np.zeros((nVarLB, originalNumVar))
        Ip[np.arange(nVarLB), varsWithLowerBound] = 1

        Ipp = np.zeros((nVarUB, originalNumVar))
        Ipp[np.arange(nVarUB), varsWithUpperBound] = 1

        # Now it's easiest to assemble the A matrix vertically:
        verticalGroup1 = np.vstack(
            [self.originalLP.A_eq, self.originalLP.A_leq, -Ip, Ipp]
        )
        verticalGroup2 = np.vstack(
            [-self.originalLP.A_eq, -self.originalLP.A_leq, Ip, -Ipp]
        )
        verticalGroup3 = np.vstack(
            [np.zeros((self.originalLP.A_eq.shape[0], numSlack)), np.identity(numSlack)]
        )

        newA = np.hstack([verticalGroup1, verticalGroup2, verticalGroup3])

        # The overall b vector can also be assembled:
        newb = np.hstack(
            [
                self.originalLP.b_eq,
                self.originalLP.b_leq,
                -self.originalLP.lb[varsWithLowerBound],
                self.originalLP.ub[varsWithUpperBound],
            ]
        )

        # Finally, we need to transform the objective:
        newc = np.hstack([self.originalLP.c, -self.originalLP.c, np.zeros(numSlack)])

        self.transformedLP = LP(newc, newA, newb, lb=np.zeros(newA.shape[1]))
        return self.transformedLP

    def UnTransform(self, x):
        # Recall that the inputted "x" here contains the positive and negative components of the original solution followed by slack variables.
        # Thus, the original solution will be the positive component minus the negative component
        originalNumVar = len(self.originalLP.c)

        #      Positive Comp.     - Negative Comp.
        return x[:originalNumVar] - x[originalNumVar : 2 * originalNumVar]


class BoundShiftingTransformer(LP_Transformer):
    """
    This transformer builds upon the same underlying logic as the BasicTransformer but takes advantage of the fact that any variable with a lower or upper bound can be shifted or flipped to get it to align will with standard form without the need for explicit variables for the positive and negative components:

    For example, consider a variable with the following bounds:

        -3 <= x <= 10

    We can define x' = x + 3 and arrive at the following:

        0 <= x' <= 13

    Which naturally aligns with the x >= 0 constraint required by standard form. All references to x can simply be replaced by x' and no components for that variable must be added. Thus, if bounds are defined for most variables, this transformer could dramatically reduce the amount of variables to consider.
    """

    pass


class SimplifyingTransformer(LP_Transformer):
    """
    This transformer builds upon some of the same logic as the BoundShiftingTransformer but also takes advantage of the fact that some variables can be entirely removed from the problem. This is done by eliminating variables and constraints that do not directly contribute to a degree of freedom.

    For example, consider this LP:

        min x
        s.t. 3x == y + 2
             y == z/2
             0 <= z <= 5

    Since the value of x can be directly determined from the value of y and since the value of y can be directly determined from the value of z, only the z variable needs to be kept. The problem can be simplified to:

        min z/6 + 2/3
        s.t. 0 <= z <= 5

    Of course this is a very simple problem. But the approach can be generalized to eliminate p variables and p constraints from a model.

    Given an LP in standard form:

        min c^T x
        s.t. A x == b
             x >= 0

    Where there are m constraints and n variables. First we'll use Singular Value Decomposition (SVD) to transform the problem in a more natural coordinate system an to naturally eliminate any linearly dependent constraints/variables.

        A = U S V^T

    The new coordinate system can be determined as follows:

        x' = V^T x

    We can apply this transformation to the constraints as follows:

        A (V x') == b => U S x' == b

    Thus we can arrive at the following, transformed LP.

        min (c')^T x'
        s.t. A' x' == b
             x' >= 0

    Where c' = V^T c, A' = U S, THIS MIGHT NOT WORK.
    """

    pass


class RepetitiveEqualityTransformer(LP_Transformer):
    """
    This transformer works by replacing all equality constraints (A x = b) with two inequality constraints (Ax <= b and -Ax <= -b) and then merging all inequality constraints into one large A_leq, b_leq system.

    Thus this transformer takes in an lp of general form:

    min c^T x
    s.t. A_eq x == b_eq
         A_leq x <= b_leq
         lb <= x <= ub

    and transforms it into an lp of the following form (also referred to as "dual standard form"):

    min c^T x
    s.t. A_leq x <= b_leq

    Mathematically speaking, here is the matrix representation of this transformed A_leq x <= b_leq system in terms of the original A_eq, b_eq, A_leq, b_leq, lb, ub values:

    min c^T x

    s.t. ┌─     ─┐┌─ ─┐    ┌─     ─┐
         │┌─   ─┐││   │    │┌─   ─┐│
         │ A_leq ││   │    │ b_leq │
         │└─   ─┘││   │    │└─   ─┘│
         │┌─   ─┐││   │    │┌─   ─┐│
         │ A_eq  ││   │    │ b_eq  │
         │└─   ─┘││   │    │└─   ─┘│
         │┌─   ─┐││   │    │┌─   ─┐│
         │ -A_eq ││ x │ <= │ -b_eq │
         │└─   ─┘││   │    │└─   ─┘│
         │┌─   ─┐││   │    │┌─   ─┐│
         │   I'  ││   │    │  lb'  │
         │└─   ─┘││   │    │└─   ─┘│
         │┌─   ─┐││   │    │┌─   ─┐│
         │  -I'' ││   │    │ -ub'' │
         │└─   ─┘││   │    │└─   ─┘│
         └─     ─┘└─ ─┘    └─     ─┘

    Where I' and lb' are the identity matrix and lb vector with rows corresponding to "nan" lb values removed. Likewise I'' and ub'' are the identity matrix and ub vector with rows corresponding to "nan" ub values removed.
    """

    def Transform(self):
        # First, let's define I', lb', I'' and ub''
        lbNonNanIndices = np.where(~np.isnan(self.originalLP.lb))[0]
        ubNonNanIndices = np.where(~np.isnan(self.originalLP.ub))[0]

        lbp = self.originalLP.lb[lbNonNanIndices]
        ubpp = self.originalLP.ub[ubNonNanIndices]

        Ip = np.zeros((len(lbNonNanIndices), len(self.originalLP.c)))
        Ip[np.arange(len(lbNonNanIndices)), lbNonNanIndices] = 1

        Ipp = np.zeros((len(ubNonNanIndices), len(self.originalLP.c)))
        Ipp[np.arange(len(ubNonNanIndices)), ubNonNanIndices] = 1

        # Now lets assemble the new A_leq and b_leq components
        A_leq = np.vstack(
            [
                self.originalLP.A_leq,
                self.originalLP.A_eq,
                -self.originalLP.A_eq,
                Ip,
                -Ipp,
            ]
        )
        b_leq = np.hstack(
            [
                self.originalLP.b_leq,
                self.originalLP.b_eq,
                -self.originalLP.b_eq,
                lbp,
                -ubpp,
            ]
        )

        self.transformedLP = LP(c=self.originalLP.c, A_leq=A_leq, b_leq=b_leq)
        return self.transformedLP

    def UnTransform(self, x):
        """
        Since no transformations are done to the space of x variables. The transformation for this transformer will be simply "x".
        """
        return x
