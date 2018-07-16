#include <vector>
#include <gtest/gtest.h>
#include "gtest_utils.h"
#include "../cl_utils.h"


using namespace cl;
using namespace std;

class CLUtilsTest : public testing::TestWithParam<Set *>
{
protected:
    Set *set;
    std::vector<uint> input, output;
    size_t size;
    void SetUp()
    {
        set = GetParam();
        size = 1280 * 720;
        input.resize(size, (uint) 0x12345678);
    }
    void TearDown()
    {
    }
};

TEST_P(CLUtilsTest, writeRead)
{
    BufferT<uint> buf(set, size);
    buf.write(input);
    buf.read(output);
    EXPECT_TRUE(vectorMatch(input, output));
}

#ifdef KERNEL_FILL
TEST_P(CLUtilsTest, kernelFill)
{
    BufferT<uint> buf(set, size);
    buf.write(input);
    FillKernel fk;
    fk(set, buf, size * sizeof(uint));
    buf.read(output);
    EXPECT_TRUE(vectorMatch(output, (uint) 0));
}
#endif

TEST_P(CLUtilsTest, bufferTFill)
{
    BufferT<uint> buf(set, size);
    buf.write(input);
    buf.fill();
    buf.read(output);
    EXPECT_TRUE(vectorMatch(output, (uint) 0));
}


INSTANTIATE_TEST_CASE_P(OpenCL, CLUtilsTest, ::testing::ValuesIn(CLEnvironment::getSets()), clDeviceName());
