#include "../hough.h"

int main()
{
    cl::Set set;
    set.initializeDefault("CUDA");
    houghTest(&set);
}
