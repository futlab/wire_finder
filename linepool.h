#ifndef LINEPOOL_H
#define LINEPOOL_H

#include <vector>
#include <eigen3/Eigen/Core>

#include "defs.h"

using namespace Eigen;

struct HypoLine
{
    Vector3d direction;
    Vector2d position;
    double p;
    short leftA, leftB;
    short rightA, rightB;
};

class LinePool
{
private:
    std::vector<HypoLine> pool_;
public:
    LinePool();
};

#endif // LINEPOOL_H
