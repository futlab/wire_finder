#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "defs.h"
#include "cameramodel.h"

using namespace Eigen;

template<typename real>
struct KalmanLine
{
	typedef Matrix<real, 3, 1> Vector3;
	typedef Matrix<real, 2, 1> Vector2;
	typedef Matrix<real, 4, 1> Vector4;
	typedef Matrix<real, 4, 4> Matrix4;
	typedef Matrix<real, 2, 2> Matrix2;
	Vector2 position;
	Vector2 direction;
	Matrix4 P;

	KalmanLine(const CameraModel<real> &camera, const LineV & left, const LineV & right, unsigned int value)
	{
		// Calculate position
		float leftX = left.b + (real)left.a / 32768.0f * camera.leftCenterY();
		float rightX = right.b + (real)right.a / 32768.0f * camera.rightCenterY();
		camera.pointByScreen(position, leftX, rightX);

		// Calculate direction
		Vector3 d;
		camera.pointByScreen(d, left.b, right.b, 0);
		d[0] -= position[0];
		d[2] -= position[1];
		direction << d[0] / d[1], d[2] / d[1];
	}
	KalmanLine(const Vector2 &position, const Vector2 &direction, real disp = 0) : position(position), direction(direction), P(Matrix4::Identity() * disp) {}
	KalmanLine(const Vector4 &x, real disp = 0) : P(Matrix4::Identity() * disp) { setX(x); }
	inline void setX(const Vector4 &x) { position = x.block<2, 1>(0, 0); direction = x.block<2, 1>(2, 0); }
	inline Vector4 getX() const { Vector4 r; r << position, direction; return r; }
	inline void getF(Matrix4 &F, const Matrix<real, 3, 3> &irm, const Vector3 &p, const Vector3 &d) const
	{
		Matrix2 irm2; irm2 << irm(0, 0), irm(0, 2), irm(2, 0), irm(2, 2);
		Matrix<real, 1, 2> irv(irm(1, 0), irm(1, 2));
		Vector2 d2(d[0], d[2]);
		const real d1 = d[1];
		// d pos / d pos
		F.block<2, 2>(0, 0) = irm2 - direction * irv;
		/*
		F(0, 0) = irm(0, 0) - irm(1, 0) * direction[0];
		F(0, 1) = irm(0, 2) - irm(1, 2) * direction[0];
		F(1, 0) = irm(2, 0) - irm(1, 0) * direction[1];
		F(1, 1) = irm(2, 2) - irm(1, 2) * direction[1];*/

		// d dir / d dir
		F.block<2, 2>(2, 2) = (irm2 * d1 - d2 * irv) / (d1 * d1);
		/*
		F(2, 2) = (irm(0, 0) * d(1) - irm(1, 0) * d(0)) / (d[1] * d[1]);
		F(2, 3) = (irm(0, 2) * d(1) - irm(1, 2) * d(0)) / (d[1] * d[1]);
		F(3, 2) = (irm(2, 0) * d(1) - irm(1, 0) * d(2)) / (d[1] * d[1]);
		F(3, 3) = (irm(2, 2) * d(1) - irm(1, 2) * d(2)) / (d[1] * d[1]);*/

		// d pos / d dir
		F.block<2, 2>(0, 2) = -p[1] * F.block<2, 2>(2, 2);
		/*
		F(0, 2) = p[1] * (irm(1, 0) * d[0] - irm(0, 0) * d[1]) / (d[1] * d[1]);
		F(1, 2) = p[1] * (irm(1, 0) * d[2] - irm(2, 0) * d[1]) / (d[1] * d[1]);*/

		// d dir / d pos
		F.block<2, 2>(2, 0) = Matrix2::Zero();
	}
	void predict(const Matrix<real, 3, 3> &invertedRotation, const Matrix<real, 3, 1> &translation)
	{
		Vector3 p(position[0], 0, position[1]);
		p = invertedRotation * (p - translation);
		auto d = invertedRotation * Vector3(direction[0], 1, direction[1]);
		direction << 
			d[0] / d[1], 
			d[2] / d[1];
		position <<
			p[0] - p[1] * direction[0],
			p[2] - p[1] * direction[1];
		Matrix4 F;
		getF(F, invertedRotation, p, d);
		P = F * P * F.transpose();
		// P += Q;
	}
	void correct(const CameraModel<real> &camera, const LineV & left, const LineV & right, unsigned int value)
	{
		Vector2 pos;
		// Calculate position
		float leftX = left.b + (real)left.a / 32768.0f * camera.leftCenterY();
		float rightX = right.b + (real)right.a / 32768.0f * camera.rightCenterY();
		camera.pointByScreen(pos, leftX, rightX);

		Vector3 dir;
		// Calculate direction
		camera.pointByScreen(dir, left.b, right.b, 0);
		dir[0] -= pos[0];
		dir[2] -= pos[1];
		dir.normalize();


		Vector5 x, z; 
		x << position, direction;
		z << pos, dir;

	}
};
