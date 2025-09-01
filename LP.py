from scipy.sparse import csr_matrix, save_npz, load_npz
import numpy as np
import tarfile, tempfile, os, warnings

class LP:
    """
    A class to house a Linear Program of the following form:

    min c^T x
    s.t. A_eq x = b_eq
         A_leq x <= b_leq
         lb <= x <= ub
    """
    def AssertValidFormatting(self):
        """
        Validates the formatting and consistency of the linear program components.
        
        Ensures that:
        - If A_eq is provided, b_eq must also be provided (and vice versa)
        - If A_leq is provided, b_leq must also be provided (and vice versa)
        - All matrices and vectors have compatible dimensions
        - All constraint matrices are 2D and constraint vectors are 1D
        - Variable bounds have the correct dimensions
        
        Returns:
            int: The number of variables in the linear program.
            
        Raises:
            AssertionError: If any formatting or dimension consistency checks fail.
        """
        assert (self.A_eq is not None) == (self.b_eq is not None), "If you provide A_eq or b_eq you must provide both"
        assert (self.A_leq is not None) == (self.b_leq is not None), "If you provide A_leq or b_leq you must provide both"

        numVar = len(self.c)
        if self.A_eq is not None:
            assert len(self.A_eq.shape) == 2, "A_eq must be a two-dimensional matrix"
            assert len(self.b_eq.shape) == 1, "b_eq must be a one-dimensional array"

            numEqConstr,nVar = self.A_eq.shape
            assert nVar == numVar, "c and A_leq have conflicting numbers of variables"
            assert numEqConstr == len(self.b_eq), "A_eq and b_eq must indicate the same number of constraints"

        if self.A_leq is not None:
            assert len(self.A_leq.shape) == 2, "A_leq must be a two-dimensional matrix"
            assert len(self.b_leq.shape) == 1, "b_leq must be a one-dimensional array"

            numLeqConstr,nVar = self.A_leq.shape
            assert nVar == numVar, "c and A_leq have conflicting numbers of variables"
            assert numLeqConstr == len(self.b_leq), "A_leq and b_leq must indicate the same number of constraints"

        if self.lb is not None:
            assert len(self.lb.shape) == 1, "lb must be a one-dimensional array"
            assert numVar == len(self.lb), "c and lb have conflicting numbers of variables"


        if self.ub is not None:
            assert len(self.ub.shape) == 1, "ub must be a one-dimensional array"
            assert numVar == len(self.ub), "c and ub have conflicting numbers of variables"

        return numVar

    def __init__(self,c:np.array,A_eq:csr_matrix=None,b_eq:np.array=None,A_leq:csr_matrix=None,b_leq:np.array=None,lb:np.array=None,ub:np.array=None):
        """
        Initialize a Linear Program instance.
        
        Creates a linear program of the form:
            min c^T x
            s.t. A_eq x = b_eq
                 A_leq x <= b_leq
                 lb <= x <= ub
        
        Args:
            c (np.array): Objective function coefficients (1D array).
            A_eq (csr_matrix, optional): Equality constraint matrix. Defaults to None.
            b_eq (np.array, optional): Equality constraint right-hand side. Defaults to None.
            A_leq (csr_matrix, optional): Inequality constraint matrix. Defaults to None.
            b_leq (np.array, optional): Inequality constraint right-hand side. Defaults to None.
            lb (np.array, optional): Lower bounds on variables. Defaults to None (no bounds).
            ub (np.array, optional): Upper bounds on variables. Defaults to None (no bounds).
            
        Note:
            If A_eq/b_eq or A_leq/b_leq are not provided, empty constraint matrices will be created.
            If lb/ub are not provided, they will be set to NaN (indicating no bounds).
        """
        self.c = c
        self.A_eq = A_eq
        self.A_leq = A_leq
        self.b_eq = b_eq
        self.b_leq = b_leq
        self.lb = lb
        self.ub = ub
        self.numVar = numVar = self.AssertValidFormatting()

        self.c = c

        if A_eq is not None:
            self.A_eq = A_eq
            self.b_eq = b_eq
        else:
            self.A_eq = csr_matrix((0,numVar),dtype=float)
            self.b_eq = np.zeros(0,dtype=float)

        if A_leq is not None:
            self.A_leq = A_leq
            self.b_leq = b_leq
        else:
            self.A_leq = csr_matrix((0,numVar),dtype=float)
            self.b_leq = np.zeros(0,dtype=float)

        if lb is not None:
            self.lb = lb
        else:
            self.lb = np.full((numVar,),np.nan)

        if ub is not None:
            self.ub = ub
        else:
            self.ub = np.full((numVar,),np.nan)

    def ToString(self,sparseFormat=True):
        """
        Generate a string representation of the linear program.
        
        Creates a formatted string showing the problem dimensions and all components
        of the linear program including objective coefficients, constraint matrices,
        and variable bounds.
        
        Args:
            sparseFormat (bool, optional): If True, displays sparse matrices in their
                sparse representation. If False, converts to dense arrays. Defaults to True.
                
        Returns:
            str: A formatted string representation of the linear program.
        """
        numEqConstr,numVar = self.A_eq.shape
        numLeqConstr = self.A_leq.shape[0]

        numLB = np.sum(~np.isnan(self.lb))
        numUB = np.sum(~np.isnan(self.ub))


        header = f"Linear Program with...\n\t{numVar} Variables\n\t{numEqConstr} Equality Constraints\n\t{numLeqConstr} Inequality Constraints\n\t{numLB} Lower Bounds\n\t{numUB} Upper Bounds."

        componentStrs = []
        if sparseFormat:
            componentStrs = ["c:",str(self.c),"A_eq:",str(self.A_eq),"b_eq:",str(self.b_eq),"A_leq:",str(self.A_leq),"b_leq:",str(self.b_leq),"lb:",str(self.lb),"ub:",str(self.ub)]
        else:
            componentStrs = ["c:",str(self.c),"A_eq:",str(self.A_eq.toarray()),"b_eq:",str(self.b_eq),"A_leq:",str(self.A_leq.toarray()),"b_leq:",str(self.b_leq),"lb:",str(self.lb),"ub:",str(self.ub)]
        
        componentStrs = "\n".join(componentStrs)
        return f"{header}\n\n{componentStrs}"

    def __str__(self):
        """
        Return a string representation of the linear program.
        
        Returns:
            str: String representation using default sparse format.
        """
        return self.ToString()

    def __repr__(self):
        """
        Return a string representation of the linear program for debugging.
        
        Returns:
            str: String representation of the linear program.
        """
        return str(self)
    
    def saveArray(self,arr,file):
        """
        Save a numpy array or sparse matrix to a file.
        
        Handles both dense numpy arrays (saved as .npy) and sparse CSR matrices
        (saved as .npz) with appropriate file extensions.
        
        Args:
            arr (np.ndarray or csr_matrix): The array or matrix to save.
            file (str): Base filename (without extension) where the array will be saved.
            
        Returns:
            str: The full filename (with extension) where the array was saved.
            
        Raises:
            Exception: If the array type is not recognized (not np.ndarray or csr_matrix).
        """
        if isinstance(arr,np.ndarray):
            fName = f"{file}.npy"
            np.save(fName,arr)
        elif isinstance(arr,csr_matrix):
            fName = f"{file}.npz"
            save_npz(fName, arr)
        else:
            raise Exception(f"Unrecognized array type: \"{type(arr)}\"")
        
        return fName
    
    def Save(self,fileName:str):
        """
        Save the linear program to a compressed tar.gz archive.
        
        Serializes all components of the linear program (objective coefficients,
        constraint matrices, bounds) into separate files and packages them into
        a single compressed archive for easy storage and transfer.
        
        Args:
            fileName (str): The name of the file to save to. If it doesn't end with
                '.tar.gz', this extension will be automatically added.
                
        Note:
            The archive contains separate files for each LP component (c, A_eq, b_eq,
            A_leq, b_leq, lb, ub) with appropriate file extensions (.npy for dense
            arrays, .npz for sparse matrices).
        """
        if not fileName.endswith(".tar.gz"):
            fileName = f"{fileName}.tar.gz"

        with tempfile.TemporaryDirectory() as tmpdirname:
            attributes = ["c","A_eq","b_eq","A_leq","b_leq","lb","ub"]
            fileNames = [None for _ in range(len(attributes))]
            for i,a in enumerate(attributes):
                fileNames[i] = self.saveArray(getattr(self,a),f"{tmpdirname}/{a}")
                

            with tarfile.open(fileName, 'w:gz') as tar:
                for f in fileNames:
                    tar.add(f,arcname=os.path.basename(f))

    @staticmethod
    def Load(fileName:str):
        """
        Load a linear program from a compressed tar.gz archive.
        
        Deserializes a linear program that was previously saved using the Save method.
        Extracts and loads all components from the archive to reconstruct the LP instance.
        
        Args:
            fileName (str): The name of the archive file to load from.
            
        Returns:
            LP: A new LP instance with the loaded data.
            
        Raises:
            Warning: If any expected component file is missing from the archive.
            
        Note:
            The method expects the archive to contain files for LP components with
            specific naming conventions (c.npy, A_eq.npz, etc.). Missing components
            will generate warnings but won't prevent loading.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            with tarfile.open(fileName, 'r:gz') as tar:
                tar.extractall(tmpdirname)

            attributes = {"c":None,"A_eq":None,"b_eq":None,"A_leq":None,"b_leq":None,"lb":None,"ub":None}

            for f in os.listdir(tmpdirname):
                fullPath = os.path.join(tmpdirname,f)
                if os.path.isfile(fullPath):
                    fileName,extension = os.path.splitext(f)
                    if fileName not in attributes:
                        continue

                    if extension == ".npy":
                        attributes[fileName] = np.load(fullPath)
                    elif extension == ".npz":
                        attributes[fileName] = load_npz(fullPath)

            for a in attributes:
                if attributes[a] is None:
                    warnings.warn(f"Not file found for component \"{a}\"")
            
            return LP(
                c=attributes["c"],
                A_eq=attributes["A_eq"],
                b_eq=attributes["b_eq"],
                A_leq=attributes["A_leq"],
                b_leq=attributes["b_leq"],
                lb=attributes["lb"],
                ub=attributes["ub"]
            )


