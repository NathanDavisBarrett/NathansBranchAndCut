#include "LP_Transformer.h"
#include "Utils.h"

LP_Transformer::LP_Transformer(LP_General* original, bool maintainOwnership) {
	this->original = original;

	//TransformLP();
	this->maintainOwnership = maintainOwnership;
}

LP_Transformer::~LP_Transformer() {
	if ((transformed != nullptr) && maintainOwnership) {
		delete transformed;
	}
}

/*void LP_Transformer::InsertSubmatrixLPT(RelevantMatrixIDs parentMatrixID, RelevantMatrixIDs subMatrixID, size_t startCol, size_t startRow, double coef = 1.0) {
	Mat parentMatrix;
	switch (parentMatrixID)
	{
	case LP_Transformer::original_A_eq:
		parentMatrix = original->A_eq;
	case LP_Transformer::original_A_leq:
		parentMatrix = original->A_leq;
	case LP_Transformer::transformed_A_eq:
		parentMatrix = transformed->A_eq;
	}

	Mat subMatrix;
	switch (subMatrixID)
	{
	case LP_Transformer::original_A_eq:
		subMatrix = original->A_eq;
	case LP_Transformer::original_A_leq:
		subMatrix = original->A_leq;
	case LP_Transformer::transformed_A_eq:
		subMatrix = transformed->A_eq;
	}

	InsertSubmatrix(parentMatrix, subMatrix * coef, startCol, startRow);
}*/

void LP_Transformer::InsertSubmatrixLPT(RelevantMatrixIDs parentMatrixID,  const Mat& subMatrix, size_t startCol, size_t startRow, double coef) {
	Mat parentMatrix;
	switch (parentMatrixID)
	{
	case LP_Transformer::original_A_eq:
		parentMatrix = original->A_eq;
	case LP_Transformer::original_A_leq:
		parentMatrix = original->A_leq;
	case LP_Transformer::transformed_A_eq:
		parentMatrix = transformed->A_eq;
	}

	InsertSubmatrix(parentMatrix, subMatrix * coef, startCol, startRow);
}

void LP_Transformer::InsertToB(const Vec& v) {
	transformed->b_eq << v;
}
void LP_Transformer::InsertToB(const Vec& v1, const Vec& v2) {
	transformed->b_eq << v1, v2;
}
void LP_Transformer::InsertToB(const Vec& v1, const Vec& v2, const Vec& v3) {
	transformed->b_eq << v1, v2, v3;
}
void LP_Transformer::InsertToB(const Vec& v1, const Vec& v2, const Vec& v3, const Vec& v4) {
	transformed->b_eq << v1, v2, v3, v4;
}

void LP_Transformer::InsertToC(const Vec& v) {
	transformed->c << v;
}
void LP_Transformer::InsertToC(const Vec& v1, const Vec& v2) {
	transformed->c << v1, v2;
}
void LP_Transformer::InsertToC(const Vec& v1, const Vec& v2, const Vec& v3) {
	transformed->c << v1, v2, v3;
}
void LP_Transformer::InsertToC(const Vec& v1, const Vec& v2, const Vec& v3, const Vec& v4) {
	transformed->c << v1, v2, v3, v4;
}

void BasicTransformer::TransformLP() {
	size_t originalNumVar = original->GetNumVar();
	size_t originalNumEqConstr = original->GetNumEqConstr();
	size_t originalNumLeqConstr = original->GetNumLeqConstr();

	const Vec originalLBCondensed = original->Get_lb_Condensed();
	const Vec originalUBCondensed = original->Get_ub_Condensed();
	size_t originalNumLb = originalLBCondensed.size();
	size_t originalNumUb = originalUBCondensed.size();

	size_t numSlack = originalNumLeqConstr + originalNumLb + originalNumUb;

	size_t newNumVar = 2 * originalNumVar + numSlack;
	size_t newNumConstr = originalNumEqConstr + numSlack;

	size_t newNumNonzeros = 2 * (original->GetNumEqNonZeros() + original->GetNumLeqNonZeros()) + 4 * originalNumVar;

	transformed = new LP_StandardForm(newNumVar, newNumConstr);
	transformed->reserve(newNumNonzeros);

	//Let's begin by inserting the positive and negative A_eq matrices.
	//	NOTE: For the rationalle behind these insertions, please refer to the docstring for this transformer in LP_Transformer.h
	InsertSubmatrixLPT(transformed_A_eq, original->Get_A_eq(), 0, 0);
	InsertSubmatrixLPT(transformed_A_eq, original->Get_A_eq(), originalNumVar, 0, -1.0);

	//Next insert the positive and negative A_leq matrices.
	InsertSubmatrixLPT(transformed_A_eq, original->Get_A_leq(), 0, originalNumEqConstr);
	InsertSubmatrixLPT(transformed_A_eq, original->Get_A_leq(), originalNumVar, originalNumEqConstr, -1.0);

	//Next insert I' as shown in the docstring for this transformer.
	Mat Ip(originalNumLb, originalNumVar);
	Ip.reserve(originalNumLb);
	size_t ii = 0;
	for (size_t i = 0; i < originalNumVar; i++) {
		if (original->HasLB(i)) {
			Ip.insert(ii, i) = 1.0;
			ii++;
		}
	}
	InsertSubmatrixLPT(transformed_A_eq, Ip, 0, originalNumEqConstr + originalNumLeqConstr, -1.0);
	InsertSubmatrixLPT(transformed_A_eq, Ip, originalNumVar, originalNumEqConstr + originalNumLeqConstr);


	//Next insert I'' as shown in the docstring for this transformer.
	Mat Ipp(originalNumLb, originalNumVar);
	Ipp.reserve(originalNumUb);
	ii = 0;
	for (size_t i = 0; i < originalNumVar; i++) {
		if (original->HasUB(i)) {
			Ipp.insert(ii, i) = 1.0;
			ii++;
		}
	}
	InsertSubmatrixLPT(transformed_A_eq, Ipp, 0, originalNumEqConstr + originalNumLeqConstr + originalNumLb);
	InsertSubmatrixLPT(transformed_A_eq, Ipp, originalNumVar, originalNumEqConstr + originalNumLeqConstr + originalNumLb, -1.0);

	//Finally, insert I as shown in the docstring for this transformer.
	Mat I(numSlack, numSlack);
	I.reserve(numSlack);
	for (size_t i = 0; i < numSlack; i++) {
		I.insert(i, i) = 1.0;
	}
	InsertSubmatrixLPT(transformed_A_eq, I, 2 * originalNumVar, originalNumEqConstr);

	//Next up assemble the new "b" vector
	InsertToB(original->Get_b_eq(), original->Get_b_leq(), -1.0 * originalLBCondensed, originalUBCondensed);

	//Lastly, assemble the new objective vector
	const Vec co = original->Get_c();
	InsertToC(co, -1.0 * co, Vec::Zero(numSlack));
}

Vec BasicTransformer::TransformSolution(Vec& originalSolution) {
	size_t originalNumVar = original->GetNumVar();
	size_t transformedNumVar = transformed->GetNumVar();
	Vec transformedSolution(transformedNumVar);

	//Step 1, split the original solution between xp and xn
	//	Split into two for loops to prevent cache misses.
	for (size_t i = 0; i < originalSolution.size(); i++) {
		double vi = originalSolution(i);
		if (vi >= 0) {
			transformedSolution(i) = vi;
		}
		else {
			transformedSolution(i) = 0;
		}
	}
	for (size_t i = 0; i < originalSolution.size(); i++) {
		size_t ii = i + originalNumVar;
		double vi = originalSolution(i);
		if (vi >= 0) {
			transformedSolution(ii) = 0;
		}
		else {
			transformedSolution(ii) = -vi;
		}
	}

	//Step 2, Compute slack values.
	size_t base = original->GetNumEqConstr();

}