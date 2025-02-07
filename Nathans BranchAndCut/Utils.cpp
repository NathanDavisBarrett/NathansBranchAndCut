#include "Utils.h"

void InsertSubmatrix(Mat& parentMatrix, Mat subMatrix, size_t startCol, size_t startRow) {
	for (int ci = 0; ci < subMatrix.cols(); ++ci) { //Iterate over the columns
		for (Mat::InnerIterator it(subMatrix,ci); it; ++it) { //Iterate the non-zero elements of this column.
			size_t ri = it.row();
			double val = it.value();

			parentMatrix.insert(ri, ci) = val;
		}
	}
}