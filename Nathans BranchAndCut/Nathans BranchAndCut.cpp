// Nathans BranchAndCut.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "VecMatHeader.h"
#include "LP.h"
#include "LP_Transformer.h"

int main()
{
    Mat A_eq(5, 5);
    Vec b_eq(5);

    Mat A_leq(5, 5);
    Vec b_leq(5);

    Vec lb(5);
    Vec ub(5);
    Vec c(5);

    LP_General genLP(A_eq, b_eq, A_leq, b_leq, lb, ub, c);

    BasicTransformer trans = LP_Transformer::Create(&genLP, true);


}


