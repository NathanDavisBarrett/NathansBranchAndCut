#include "LP.h"
#include "Exception.h"
#include <cfloat>
#include <cmath>

#define INCOMPATIBLE_SIZE_MESSAGE "The inputted vectors and/or matrices are of inconsistent size."

LP_General::LP_General(size_t numVar, size_t numEqConstr, size_t numLeqConstr) {
	A_eq = Mat(numEqConstr, numVar);
	b_eq = Vec(numEqConstr);

	A_leq = Mat(numLeqConstr, numVar);
	b_leq = Vec(numLeqConstr);

	lb = Vec(numVar);
	ub = Vec(numVar);

	c = Vec(numVar);
}

LP_General::LP_General(Mat A_eq, Vec b_eq, Mat A_leq, Vec b_leq, Vec lb, Vec ub, Vec c) {
	//First item of buisness, determine the number of variables and assert that it is consistent across all the inputs.
	size_t numVar = A_eq.cols();

	if (numVar != A_leq.cols()) {
		throw ModelFormattingException(INCOMPATIBLE_SIZE_MESSAGE);
	}
	if (numVar != lb.size()) {
		throw ModelFormattingException(INCOMPATIBLE_SIZE_MESSAGE);
	}
	if (numVar != ub.size()) {
		throw ModelFormattingException(INCOMPATIBLE_SIZE_MESSAGE);
	}
	if (numVar != c.size()) {
		throw ModelFormattingException(INCOMPATIBLE_SIZE_MESSAGE);
	}

	if (A_eq.rows() != b_eq.size()) {
		throw ModelFormattingException(INCOMPATIBLE_SIZE_MESSAGE);
	}
	if (A_leq.rows() != b_leq.size()) {
		throw ModelFormattingException(INCOMPATIBLE_SIZE_MESSAGE);
	}


	this->A_eq = A_eq;
	this->A_leq = A_leq;
	
	this->b_eq = b_eq;
	this->b_leq = b_leq;
	
	this->ub = ub;
	this->lb = lb;

	this->c = c;
}

size_t LP_General::GetNumVar() {
	return c.size();
}
size_t LP_General::GetNumEqConstr() {
	return A_eq.rows();
}
size_t LP_General::GetNumLeqConstr() {
	return A_leq.rows();
}

size_t LP_General::GetNumEqNonZeros() {
	return A_eq.nonZeros();
}
size_t LP_General::GetNumLeqNonZeros() {
	return A_leq.nonZeros();
}

size_t LP_General::GetNumLB() {
	/**
	*  A function to determine the number of non-NaN lower bounds present.
	*/
	size_t numLB = 0;
	for (size_t i = 0; i < GetNumVar(); i++) {
		if (std::isnan(lb(i))) {
			numLB++;
		}
	}
	return numLB;
}
size_t LP_General::GetNumUB() {
	/**
	*  A function to determine the number of non-NaN lower bounds present.
	*/
	size_t numUB = 0;
	for (size_t i = 0; i < GetNumVar(); i++) {
		if (std::isnan(ub(i))) {
			numUB++;
		}
	}
	return numUB;
}


LP_StandardForm::LP_StandardForm(size_t numVar, size_t numConstr) {
	A_eq = Mat(numConstr, numVar);
	b_eq = Vec(numConstr);

	c = Vec(numVar);
}

LP_StandardForm::LP_StandardForm(Mat A_eq, Vec b_eq, Vec c) {
	//First item of buisness, determine the number of variables and assert that the dimensions of the input are consistent.
	if (A_eq.cols() != c.size()) {
		throw ModelFormattingException(INCOMPATIBLE_SIZE_MESSAGE);
	}
	if (A_eq.rows() != b_eq.size()) {
		throw ModelFormattingException(INCOMPATIBLE_SIZE_MESSAGE);
	}

	this->A_eq = A_eq;
	this->b_eq = b_eq;
	this->c = c;
}

size_t LP_StandardForm::GetNumVar() {
	return c.size();
}
size_t LP_StandardForm::GetNumConstr() {
	return b_eq.size();
}
size_t LP_StandardForm::GetNumNonZeros() {
	return A_eq.nonZeros();
}

void LP_StandardForm::reserve(size_t numNonZeros) {
	A_eq.reserve(numNonZeros);
}