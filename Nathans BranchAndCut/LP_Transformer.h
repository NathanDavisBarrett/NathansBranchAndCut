#ifndef LP_TRANSFORMER_H
#define LP_TRANSFORMER_H

#include "LP.h"

#include "VecMatHeader.h"

class LP_Transformer {
	/**
	*  A class to handle the transformation of a general LP to a standard form LP.
	*/
protected:
	LP_General* original;
	LP_StandardForm* transformed;
	virtual void TransformLP() = 0;
    bool maintainOwnership;

public:
	LP_Transformer(LP_General* original, bool maintainOwnership = true);

	virtual Vec TransformSolution(Vec originalSolution) = 0;
	virtual Vec UnTransformSolution(Vec transformedSolution) = 0;

    ~LP_Transformer();
};

class BasicTransformer : public LP_Transformer {
	/**
	*  This transformer works by reading the inputted LP as is and accomplishing the transformation by taking advantage of the fact that any inequality
    *
    *     a x <= b
    *
    *  Can be transformed into the appropriate form by introducing a slack variable.
    *
    *     a x + s == b
    *     s >= 0
    *
    *  However, since the original "x" variables can be positive or negative, we also need to define positive and negative components for each variable.
    *
    *     x = xp - xn
    *     xp >= 0
    *     xn >= 0
    *
    *  Thus, any equality constraint given as a x == b should be reformulated as a xp - a xn == b.
    *  Any inequality constraint given as a x <= b should be reformulated as a xp - a xn + s == b.
    *
    *  Thus, the final form of the transformed LP A x == b constraint is as follows:
    *  ┌─                                  ─┐ ┌─         ─┐    ┌─         ─┐
    *  │ ┌─       ─┐ ┌─       ─┐ ┌─      ─┐ │ │  ┌─    ─┐ │    │ ┌─     ─┐ │
    *  │     A_eq      - A_eq        0      │ │     xp    │    │    b_eq   │
    *  │ └─       ─┘ └─       ─┘ └─      ─┘ │ │           │    │ └─     ─┘ │
    *  | ┌─       ─┐ ┌─       ─┐ ┌─      ─┐ │ │  └─    ─┘ │    │ ┌─     ─┐ │
    *  │    A_leq      - A_leq              │ │  ┌─    ─┐ │    │   b_leq   │
    *  │ └─       ─┘ └─       ─┘            │ │     xn    │ == │ └─     ─┘ │
    *  | ┌─       ─┐ ┌─       ─┐            │ │           │    │ ┌─     ─┐ │
    *  |     -I'          I'         I      │ │  └─    ─┘ │    │    -lb    │
    *  │ └─       ─┘ └─       ─┘            │ │  ┌─    ─┐ │    │ └─     ─┘ │
    *  | ┌─       ─┐ ┌─       ─┐            │ │     s     │    │ ┌─     ─┐ │
    *  |      I''        -I''               │ │           │    │     ub    │
    *  │ └─       ─┘ └─       ─┘ └─      ─┘ │ │  └─    ─┘ │    │ └─     ─┘ │
    *  └─                                  ─┘ └─         ─┘    └─         ─┘
    *
    *  Where I' is the identity matrix with rows removed that have their "1" value in a column that represents a variable that does not have a lower bound. Likewise, I'' is the identity matrix with rows removed that have their "1" value in a column that represents a variable that does not have an upper bound.
	*/
    void TransformLP();
    Vec TransformSolution(Vec originalSolution);
    Vec UnTransformSolution(Vec transformedSolution);
};


#endif // !LP_TRANSFORMER_H
