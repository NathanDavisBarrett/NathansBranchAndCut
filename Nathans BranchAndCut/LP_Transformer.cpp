#include "LP_Transformer.h"
#include "Utils.h"

LP_Transformer::LP_Transformer(LP_General* original, bool maintainOwnership) {
	this->original = original;

	TransformLP();
	this->maintainOwnership = maintainOwnership;
}

LP_Transformer::~LP_Transformer() {
	if ((transformed != nullptr) && maintainOwnership) {
		delete transformed;
	}
}

void LP_Transformer::InsertSubmatrixLPT(RelevantMatrixIDs parentMatrixID, RelevantMatrixIDs subMatrixID, size_t startCol, size_t startRow, double coef = 1.0) {
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
}

void BasicTransformer::TransformLP() {
	size_t originalNumVar = original->GetNumVar();
	size_t originalNumEqConstr = original->GetNumEqConstr();
	size_t originalNumLeqConstr = original->GetNumLeqConstr();
	size_t originalNumLb = original->GetNumLB();
	size_t originalNumUb = original->GetNumUB();

	size_t numSlack = originalNumLeqConstr + originalNumLb + originalNumUb;

	size_t newNumVar = 2 * originalNumVar + numSlack;
	size_t newNumConstr = originalNumEqConstr + numSlack;

	size_t newNumNonzeros = 2 * (original->GetNumEqNonZeros() + original->GetNumLeqNonZeros()) + 4 * originalNumVar;

	transformed = new LP_StandardForm(newNumVar, newNumConstr);
	transformed->reserve(newNumNonzeros);

	//Let's begin by inserting the positive and negative A_eq matrices.
	InsertSubmatrixLPT(transformed_A_eq, original_A_eq, 0, 0);
	InsertSubmatrixLPT(transformed_A_eq, original_A_eq, originalNumVar, 0, -1.0);

	//Next insert the positive and negative A_leq matrices.
	InsertSubmatrixLPT(transformed_A_eq, original_A_leq, 0, originalNumEqConstr);
	InsertSubmatrixLPT(transformed_A_eq, original_A_leq, originalNumVar, originalNumEqConstr, -1.0);



}