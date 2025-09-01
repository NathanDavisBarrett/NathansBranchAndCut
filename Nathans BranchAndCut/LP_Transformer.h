#ifndef LP_TRANSFORMER_H
#define LP_TRANSFORMER_H

#include "VecMatHeader.h"
#include "LP.h"

class LP_Transformer {
	/**
	*  A class to handle the transformation of a general LP to a standard form LP.
    * 
    *  Note that, due to the two-step nature of initializing a transformer, a transformer should be initialized using the following syntax:
    * 
    *     Child_Transformer trans = LP_Transformer::Create<Child_Transformer>(generalLP, maintOwnership);
	*/
protected:
	LP_General* original;
	LP_StandardForm* transformed;
	virtual void TransformLP() = 0;
    bool maintainOwnership;

    LP_Transformer(LP_General* original, bool maintainOwnership = true);

public:
	virtual Vec TransformSolution(const Vec& originalSolution) = 0;
	virtual Vec UnTransformSolution(const Vec& transformedSolution) = 0;

    enum RelevantMatrixIDs {
        original_A_eq,
        original_A_leq,
        transformed_A_eq
    };

    //void InsertSubmatrixLPT(RelevantMatrixIDs parentMatrixID, RelevantMatrixIDs subMatrixID, size_t startCol, size_t startRow, double coef);
    void InsertSubmatrixLPT(RelevantMatrixIDs parentMatrixID, const Mat& subMatrix, size_t startCol, size_t startRow, double coef = 1.0);

    //template<typename... Vec>
    //void InsertToB(const Vec&... vecs);
    void InsertToB(const Vec& v);
    void InsertToB(const Vec& v1, const Vec& v2);
    void InsertToB(const Vec& v1, const Vec& v2, const Vec& v3);
    void InsertToB(const Vec& v1, const Vec& v2, const Vec& v3, const Vec& v4);

    void InsertToC(const Vec& v);
    void InsertToC(const Vec& v1, const Vec& v2);
    void InsertToC(const Vec& v1, const Vec& v2, const Vec& v3);
    void InsertToC(const Vec& v1, const Vec& v2, const Vec& v3, const Vec& v4);

    //template<typename... Vec>
    //void InsertToC(const Vec&... vecs);

    ~LP_Transformer();

    template <typename T>
    static T Create(LP_General* original, bool maintainOwnership = true) {
        T obj(original, maintainOwnership);
        obj.TransformLP();
        return obj;
    }
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
protected:
    void TransformLP();
public:
    Vec TransformSolution(Vec& originalSolution);
    Vec UnTransformSolution(Vec& transformedSolution);
};


#endif // !LP_TRANSFORMER_H
