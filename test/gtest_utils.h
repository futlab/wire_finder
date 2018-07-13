#pragma once
#include <forward_list>
#include <memory>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include "../cl_utils.h"

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

class CLEnvironment : public ::testing::Environment
{
private:
    static std::forward_list<cl::Set> sets;
    static bool initialized;
public:
    static std::vector<cl::Set *> getSets();
};

struct clDeviceName
{
    std::string operator()( const testing::TestParamInfo<cl::Set *>& info ) const
    {
        //auto location = static_cast<Location>(info.param);
        std::string s = info.param->devices[0].getInfo<CL_DEVICE_NAME>();
        std::replace(s.begin(), s.end(), ' ', '_');
        return s;//
    }
};
