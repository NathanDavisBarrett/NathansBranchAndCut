from LP import LP

from scipy.sparse import csr_matrix, save_npz, load_npz
import numpy as np
import tarfile, tempfile, os, warnings

class MILP(LP):
    """
    A MILP is just an LP with additional integrality constraints.
    """
    def __init__(self,c:np.array,integerVars:np.array,A_eq:csr_matrix=None,b_eq:np.array=None,A_leq:csr_matrix=None,b_leq:np.array=None,lb:np.array=None,ub:np.array=None):
        super().__init__(c,A_eq,b_eq,A_leq,b_leq,lb,ub)
        self.integerVars = integerVars


    def ToString(self,sparseFormat=True):
        numEqConstr,numVar = self.A_eq.shape
        numLeqConstr = self.A_leq.shape[0]

        numLB = np.sum(~np.isnan(self.lb))
        numUB = np.sum(~np.isnan(self.ub))

        numInteger = np.sum(self.integerVars)

        header = f"Mixed-Integer Linear Program with...\n\t{numVar} Variables\n\t{numEqConstr} ({numInteger} integer) Equality Constraints\n\t{numLeqConstr} Inequality Constraints\n\t{numLB} Lower Bounds\n\t{numUB} Upper Bounds."

        componentStrs = []
        if sparseFormat:
            componentStrs = ["Integer Variables:",np.where(self.integerVars)[0],"c:",str(self.c),"A_eq:",str(self.A_eq),"b_eq:",str(self.b_eq),"A_leq:",str(self.A_leq),"b_leq:",str(self.b_leq),"lb:",str(self.lb),"ub:",str(self.ub)]
        else:
            componentStrs = ["Integer Variables:",np.where(self.integerVars)[0],"c:",str(self.c),"A_eq:",str(self.A_eq.toarray()),"b_eq:",str(self.b_eq),"A_leq:",str(self.A_leq.toarray()),"b_leq:",str(self.b_leq),"lb:",str(self.lb),"ub:",str(self.ub)]
        
        componentStrs = "\n".join(componentStrs)
        return f"{header}\n\n{componentStrs}"
    
    def Save(self,fileName:str):
        if not fileName.endswith(".tar.gz"):
            fileName = f"{fileName}.tar.gz"

        with tempfile.TemporaryDirectory() as tmpdirname:
            attributes = ["integerVars","c","A_eq","b_eq","A_leq","b_leq","lb","ub"]
            fileNames = [None for _ in range(len(attributes))]
            for i,a in enumerate(attributes):
                fileNames[i] = self.saveArray(getattr(self,a),f"{tmpdirname}/{a}")
                

            with tarfile.open(fileName, 'w:gz') as tar:
                for f in fileNames:
                    tar.add(f,arcname=os.path.basename(f))

    @staticmethod
    def Load(fileName:str):
        with tempfile.TemporaryDirectory() as tmpdirname:
            with tarfile.open(fileName, 'r:gz') as tar:
                tar.extractall(tmpdirname)

            attributes = {"integerVars":None,"c":None,"A_eq":None,"b_eq":None,"A_leq":None,"b_leq":None,"lb":None,"ub":None}

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
            
            return MILP(
                integerVars=attributes["integerVars"],
                c=attributes["c"],
                A_eq=attributes["A_eq"],
                b_eq=attributes["b_eq"],
                A_leq=attributes["A_leq"],
                b_leq=attributes["b_leq"],
                lb=attributes["lb"],
                ub=attributes["ub"]
            )


