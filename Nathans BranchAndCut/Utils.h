#ifndef UTILS_H
#define UTILS_H

#include "VecMatHeader.h"

#define SP_IDENTITY(name, deg) Mat name(deg, deg); name.setIdentity();

void InsertSubmatrix(Mat& parentMatrix, const Mat& subMatrix, size_t startCol, size_t startRow);

#endif // !UTILS_H
