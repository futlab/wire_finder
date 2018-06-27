#ifndef LINEPOOL_H
#define LINEPOOL_H

#include <vector>
#include <Eigen/Core>

#include "defs.h"

using namespace Eigen;

struct HypoLine
{
    Vector3f direction;
    Vector2f position;
    float p;
    short leftA, leftB;
    short rightA, rightB;
};

class LinePool
{
private:
    std::vector<HypoLine> pool_;
	float baseline, lfx, lcx, lfy, lcy, rfx, rcx, rfy, rcy;
	void getHypoLine(HypoLine &out, const LineV &left, const LineV &right);
public:
	void onLines(const std::vector<LineV> &leftLines, const std::vector<LineV> &rightLines, const std::vector<unsigned int> &compare);
	void setCameraParams(float baseline, float lfx, float lcx, float lfy, float lcy, float rfx, float rcx, float rfy, float rcy);
	void show(float leftHFov, float rightHFov);
    LinePool();
};

#endif // LINEPOOL_H
