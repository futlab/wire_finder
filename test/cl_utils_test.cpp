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

TEST_P(CLUtilsTest, atomicCmpXchg)
{
    BufferT<uint> buf(set, 1);
    buf.fill();
    string source = "__kernel void inc(__global volatile uint *flag) { \
            while( atomic_cmpxchg(flag, get_global_id(0), get_global_id(0) + 1) != get_global_id(0)); \
            }\n";
    auto p = set->buildProgramFromSource(source);
    Kernel kernel(p, "inc");
    kernel.setArg(0, buf);
    set->queue.enqueueNDRangeKernel(kernel, NDRange(), NDRange(64));
    buf.read(output);
    EXPECT_EQ(output.size(), 1);
    EXPECT_EQ(output[0], 64);
}

INSTANTIATE_TEST_CASE_P(OpenCL, CLUtilsTest, ::testing::ValuesIn(CLEnvironment::getSets()), clDeviceName());
