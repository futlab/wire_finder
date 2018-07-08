#pragma once

#include <Eigen/Core>

using namespace Eigen;

template<typename real>
class CameraModel
{
private:
	real baseline, lfx, lcx, lfy, lcy, rfx, rcx, rfy, rcy;
	real lfx_fy, rfx_fy;
public:
	inline real leftCenterY() { return lcy; }
	inline real rightCenterY() { return rcy; }
	typedef Matrix<real, 3, 1> Vector3;
	typedef Matrix<real, 2, 1> Vector2;
	void pointByScreen(Vector2 &result, real lx, real rx) const
	{
		lx -= lcx; rx -= rcx;
		real x = baseline * lx * rfx / (lx * rfx - rx * lfx);
		real z = x * lfx / lx;
		result << x, z;
	}
	void pointByScreen(Vector3 &result, real lx, real rx, real ly) const
	{
		lx -= lcx; rx -= rcx; ly -= lcy;
		real x = baseline * lx * rfx / (lx * rfx - rx * lfx);
		real z = x * lfx / lx;
		real y = ly / lfy * z;
		result << x, y, z;
	}
	inline Vector2 pointByScreen(real lx, real rx) const { Vector2 r; pointByScreen(r, lx, rx); return r; }
	inline Vector3 pointByScreen(real lx, real rx, real ly) const { Vector3 r; pointByScreen(r, lx, rx, ly); return r; }
	void setParams(real baseline, real lfx, real lcx, real lfy, real lcy, real rfx, real rcx, real rfy, real rcy)
	{
		this->baseline = baseline;
		this->lfx = lfx;
		this->lcx = lcx;
		this->lfy = lfy;
		this->lcy = lcy;
		this->rfx = rfx;
		this->rcx = rcx;
		this->rfy = rfy;
		this->rcy = rcy;
		lfx_fy = lfx / lfy;
		rfx_fy = rfx / rfy;
	}
	void predict(const Quaternion<real> &r, const Vector3 &t, Vector3 &v)
	{
		auto irm = r.inverse().toRotationMatrix();
		v = irm * (v - t);
	}
	template<bool right>
	void projectLine(Vector2 &line, const Vector2 &position, const Vector2 &direction)
	{
		real b = right ?
			(position[0] - baseline) / position[1] * rfx + rcx :
			 position[0]             / position[1] * lfx + lcx;
		auto end = position + direction;
		
		real xd = right ?
			rcx + (end[0] - baseline) / end[1] * rfx :
			lcx +  end[0]             / end[1] * lfx;
		real yd = (right ? rfy : lfy)  / end[1];
		line << (xd - b) / yd, b;
	}
	template<bool right>
	void projectLineDiff(Matrix<real, 2, 4> &H, const Vector2 &position, const Vector2 &direction)
	{
		const real x = position[0] - (right ? baseline : 0);
		// db / dx
		H(1, 0) = (right ? rfx : lfx) / position[1];
		H(1, 1) = -(right ? rfx : lfx) * x / (position[1] * position[1]);
		H.block<1, 2>(1, 2) = Matrix<real, 1, 2>::Zero();
		// da / dx
		const auto e = position + direction;
		const real ff = (right ? rfx_fy : lfx_fy);
		H(0, 0) = -ff * direction[1] / position[1];
		H(0, 1) = ff * direction[1] * x / (position[1] * position[1]);
		H(0, 2) = ff;
		H(0, 3) = -ff * x / position[1];
	}
	template<bool right>
	inline void projectLine(real &a, real &b, const Vector2 &position, const Vector2 &direction) 
	{ 
		Vector2 line; 
		projectLine<right>(line, position, direction);
		a = line[0];
		b = line[1];
	}
	template<bool right>
	void projectPoint(Vector2 &out, const Vector3 &point)
	{
		if (right) out <<
			rcx + (point[0] - baseline) / point[2] * rfx,
			rcy + point[1] / point[2] * lfy;
		else out <<
			lcx + point[0] / point[2] * lfx,
			lcy + point[1] / point[2] * lfy;
	}
};
