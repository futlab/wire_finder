#pragma once
#include <gtest/gtest.h>
#include <Eigen/Core>

template<typename T, int rows, int cols>
testing::AssertionResult MatrixMatch(const Eigen::Matrix<T, rows, cols> &expected, const Eigen::Matrix<T, rows, cols> &actual, T eps)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
			T e = expected(i, j), a = actual(i, j);
			if (abs(e - a) >= eps) {
				return ::testing::AssertionFailure() << "expected[" << i << ", " << j << "] (" << e << ") != actual[" << i << ", " << j << "] (" << a << ")";
			}
		}
	return ::testing::AssertionSuccess();
}
