#ifndef LP_H
#define LP_H

#include "VecMatHeader.h"
#include "LP_Transformer.h"

struct LP_General {
	/**
	*  A class to house a general LP
	*/
private:
	Mat A_eq;
	Vec b_eq;

	Mat A_leq;
	Vec b_leq;

	Vec lb;
	Vec ub;

	Vec c;

public:
	LP_General(size_t numVar, size_t numEqConstr, size_t numLeqConstr);

	LP_General(Mat A_eq, Vec b_eq, Mat A_leq, Vec b_leq, Vec lb, Vec ub, Vec c);

	size_t GetNumVar();
	size_t GetNumEqConstr();
	size_t GetNumLeqConstr();

	size_t GetNumEqNonZeros();
	size_t GetNumLeqNonZeros();

	size_t GetNumLB();
	size_t GetNumUB();

	Mat::InnerIterator A_eq_Itr(size_t coli);
	Mat::InnerIterator A_leq_Itr(size_t coli);

	friend class LP_Transformer;
};

struct LP_StandardForm {
	/**
	*  A class to house an LP in standard form. This is needed for most solvers.
	*  For our purposes, a standard form LP is defined as follows:
	*  
	*  min c^T x
	*  s.t. A x = b
	*       x >= 0
	* 
	*  Given a general LP, there are several ways to reformulate to a standard form LP. To accomplish this, please utilize (or create your own) LP_Transformer.
	*/
private:
	Mat A_eq;
	Vec b_eq;

	Vec c;

public:
	LP_StandardForm(size_t numVar, size_t numConstr);

	LP_StandardForm(Mat A_eq, Vec b_eq, Vec c);

	size_t GetNumVar();
	size_t GetNumConstr();
	size_t GetNumNonZeros();

	void reserve(size_t numNonZeros);
	void insert(size_t coli, size_t rowi, double value);

	friend class LP_Transformer;
};

#endif
