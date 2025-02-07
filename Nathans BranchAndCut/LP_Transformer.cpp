#include "LP_Transformer.h"

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

	//Now let's begin by loading the new A_eq matrix.

	
}