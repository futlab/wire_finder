#include <iostream>

#include "cl_utils.h"

using namespace std;

int main()
{
    CLWrapper cl;
    auto k = cl.loadKernels("kernel.cl", {"test"});
    cout << "Hello World!" << endl;
    return 0;
}
