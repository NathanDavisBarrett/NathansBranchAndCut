from LP import LP

import numpy as np
import tarfile, tempfile, os, warnings


class MILP(LP):
    """
    A Mixed-Integer Linear Program (MILP) class extending the LP class.

    Represents optimization problems where some variables are constrained to be integers
    while others can be continuous. Inherits all functionality from the LP class and
    adds integer variable constraints.

    The MILP has the form:
        min c^T x
        s.t. A_eq x = b_eq
             A_leq x <= b_leq
             lb <= x <= ub
             x[i] ∈ ℤ for i ∈ integerVars

    Attributes:
        integerVars (np.array): Boolean array indicating which variables must be integers.
        (Also inherits all attributes from LP class)
    """

    def __init__(
        self,
        c: np.array,
        integerVars: np.array,
        A_eq: np.array = None,
        b_eq: np.array = None,
        A_leq: np.array = None,
        b_leq: np.array = None,
        lb: np.array = None,
        ub: np.array = None,
    ):
        """
        Initialize a Mixed-Integer Linear Program.

        Args:
            c (np.array): Objective function coefficients.
            integerVars (np.array): Boolean array where True indicates the variable must be integer.
            A_eq (csr_matrix, optional): Equality constraint matrix. Defaults to None.
            b_eq (np.array, optional): Equality constraint RHS. Defaults to None.
            A_leq (csr_matrix, optional): Inequality constraint matrix. Defaults to None.
            b_leq (np.array, optional): Inequality constraint RHS. Defaults to None.
            lb (np.array, optional): Lower bounds on variables. Defaults to None.
            ub (np.array, optional): Upper bounds on variables. Defaults to None.
        """
        super().__init__(c, A_eq, b_eq, A_leq, b_leq, lb, ub)
        self.integerVars = integerVars

    def ToString(self, sparseFormat=True):
        """
        Generate a string representation of the mixed-integer linear program.

        Creates a formatted string showing problem dimensions, including the number
        of integer variables, and all MILP components.

        Args:
            sparseFormat (bool, optional): If True, displays sparse matrices in sparse format.
                If False, converts to dense arrays. Defaults to True.

        Returns:
            str: A formatted string representation of the MILP.
        """
        numEqConstr, numVar = self.A_eq.shape
        numLeqConstr = self.A_leq.shape[0]

        numLB = np.sum(~np.isnan(self.lb))
        numUB = np.sum(~np.isnan(self.ub))

        numInteger = np.sum(self.integerVars)

        header = f"Mixed-Integer Linear Program with...\n\t{numVar} Variables\n\t{numEqConstr} ({numInteger} integer) Equality Constraints\n\t{numLeqConstr} Inequality Constraints\n\t{numLB} Lower Bounds\n\t{numUB} Upper Bounds."

        componentStrs = []
        if sparseFormat:
            componentStrs = [
                "Integer Variables:",
                np.where(self.integerVars)[0],
                "c:",
                str(self.c),
                "A_eq:",
                str(self.A_eq),
                "b_eq:",
                str(self.b_eq),
                "A_leq:",
                str(self.A_leq),
                "b_leq:",
                str(self.b_leq),
                "lb:",
                str(self.lb),
                "ub:",
                str(self.ub),
            ]
        else:
            componentStrs = [
                "Integer Variables:",
                np.where(self.integerVars)[0],
                "c:",
                str(self.c),
                "A_eq:",
                str(self.A_eq.toarray()),
                "b_eq:",
                str(self.b_eq),
                "A_leq:",
                str(self.A_leq.toarray()),
                "b_leq:",
                str(self.b_leq),
                "lb:",
                str(self.lb),
                "ub:",
                str(self.ub),
            ]

        componentStrs = "\n".join(componentStrs)
        return f"{header}\n\n{componentStrs}"

    def Save(self, fileName: str):
        """
        Save the mixed-integer linear program to a compressed archive.

        Extends the LP save functionality to also save the integer variable constraints.
        All MILP components are serialized and packaged into a tar.gz archive.

        Args:
            fileName (str): Name of the file to save to. '.tar.gz' extension will be
                added automatically if not present.
        """
        if not fileName.endswith(".tar.gz"):
            fileName = f"{fileName}.tar.gz"

        with tempfile.TemporaryDirectory() as tmpdirname:
            attributes = [
                "integerVars",
                "c",
                "A_eq",
                "b_eq",
                "A_leq",
                "b_leq",
                "lb",
                "ub",
            ]
            fileNames = [None for _ in range(len(attributes))]
            for i, a in enumerate(attributes):
                fileNames[i] = self.saveArray(getattr(self, a), f"{tmpdirname}/{a}")

            with tarfile.open(fileName, "w:gz") as tar:
                for f in fileNames:
                    tar.add(f, arcname=os.path.basename(f))

    @staticmethod
    def Load(fileName: str):
        """
        Load a mixed-integer linear program from a compressed archive.

        Deserializes a MILP that was previously saved using the Save method.
        Reconstructs all components including integer variable constraints.

        Args:
            fileName (str): Name of the archive file to load from.

        Returns:
            MILP: A new MILP instance with the loaded data.

        Raises:
            Warning: If any expected component file is missing from the archive.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            with tarfile.open(fileName, "r:gz") as tar:
                tar.extractall(tmpdirname)

            attributes = {
                "integerVars": None,
                "c": None,
                "A_eq": None,
                "b_eq": None,
                "A_leq": None,
                "b_leq": None,
                "lb": None,
                "ub": None,
            }

            for f in os.listdir(tmpdirname):
                fullPath = os.path.join(tmpdirname, f)
                if os.path.isfile(fullPath):
                    fileName, extension = os.path.splitext(f)
                    if fileName not in attributes:
                        continue

                    if extension == ".npy":
                        attributes[fileName] = np.load(fullPath)

            for a in attributes:
                if attributes[a] is None:
                    warnings.warn(f'Not file found for component "{a}"')

            return MILP(
                integerVars=attributes["integerVars"],
                c=attributes["c"],
                A_eq=attributes["A_eq"],
                b_eq=attributes["b_eq"],
                A_leq=attributes["A_leq"],
                b_leq=attributes["b_leq"],
                lb=attributes["lb"],
                ub=attributes["ub"],
            )
