#include <gtest/gtest.h>
#include "gtest_utils.h"
#include "../hough.h"

inline int angle2tg(float a) { return int(32768.f * tanf(a)); }

void tgSumTest()
{
	int m = 0;
	float am, bm;
	for (float a = -0.4f; a < 0.4f; a += 0.003f)
		for (float b = -0.4f; b < 0.4f; b += 0.003f) {
			int at = angle2tg(a), bt = angle2tg(b);
			int div = 0x40000000 - at * bt;
			if (!div) continue;
			int rt = ((at + bt) << 15) / (div >> 15), tt = angle2tg(a + b);
			if (abs(rt - tt) > m) {
				m = abs(rt - tt);
				am = a;
				bm = b;
			}
		}
	printf("%d", m);
}

void testLinepool();

std::forward_list<cl::Set> CLEnvironment::sets;
bool CLEnvironment::initialized = false;

std::vector<cl::Set *> CLEnvironment::getSets()
{
    if (!initialized) {
        for (auto platform : cl::Set::getPlatforms())
            for (auto device : cl::Set::getDevices(platform))
                sets.emplace_front(device, CL_QUEUE_PROFILING_ENABLE);
        initialized = true;
    }
    std::vector<cl::Set *> result;
    for (auto &s : sets)
        result.push_back(&s);
    return result;
}

int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new CLEnvironment);

	int result = RUN_ALL_TESTS();
	std::cin.get();
	return result;

	testLinepool();
	tgSumTest();
    cl::Set set;
    set.initializeDefault("CUDA");
    houghTest(&set);
}
