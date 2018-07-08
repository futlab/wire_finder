#pragma once
#include <gtest/gtest.h>
#include <Eigen/Core>

template<typename T, size_t rows, size_t cols>
testing::AssertionResult MatrixMatch(const Eigen::Matrix<T, rows, cols> &expected, const Eigen::Matrix<T, rows, cols> &actual, T eps)
{
	for (size_t i = 0; i < rows; i++)
		for (size_t j = 0; j < cols; j++) {
			T e = expected(i, j), a = actual(i, j);
			if (abs(e - a) >= eps) {
				return ::testing::AssertionFailure() << "expected[" << i << ", " << j << "] (" << e << ") != actual[" << i << ", " << j << "] (" << a << ")";
			}
		}
	return ::testing::AssertionSuccess();
}
