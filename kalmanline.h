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

	KalmanLine(const CameraModel<real> &camera, const LineV & left, const LineV & right, real posP, real dirP)
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
		P = Matrix4::Identity();
		P.block<2, 2>(0, 0) *= posP;
		P.block<2, 2>(2, 2) *= dirP;
	}
	KalmanLine(const Vector2 &position, const Vector2 &direction, real disp = 0) : position(position), direction(direction), P(Matrix4::Identity() * disp) {}
	KalmanLine(const Vector4 &x, real disp = 0) : P(Matrix4::Identity() * disp) { setX(x); }
    inline void setX(const Vector4 &x) { position = x.template block<2, 1>(0, 0); direction = x.template block<2, 1>(2, 0); }
	inline Vector4 getX() const { Vector4 r; r << position, direction; return r; }
	inline void getF(Matrix4 &F, const Matrix<real, 3, 3> &irm, const Vector3 &p, const Vector3 &d) const
	{
		Matrix2 irm2; irm2 << irm(0, 0), irm(0, 2), irm(2, 0), irm(2, 2);
		const Matrix<real, 1, 2> irv(irm(1, 0), irm(1, 2));
		const Vector2 d2(d[0], d[2]);
		const real d1 = d[1];
		// d pos / d pos
        F.template block<2, 2>(0, 0) = irm2 - direction * irv;
		/*
		F(0, 0) = irm(0, 0) - irm(1, 0) * direction[0];
		F(0, 1) = irm(0, 2) - irm(1, 2) * direction[0];
		F(1, 0) = irm(2, 0) - irm(1, 0) * direction[1];
		F(1, 1) = irm(2, 2) - irm(1, 2) * direction[1];*/

		// d dir / d dir
        F.template block<2, 2>(2, 2) = (irm2 * d1 - d2 * irv) / (d1 * d1);
		/*
		F(2, 2) = (irm(0, 0) * d(1) - irm(1, 0) * d(0)) / (d[1] * d[1]);
		F(2, 3) = (irm(0, 2) * d(1) - irm(1, 2) * d(0)) / (d[1] * d[1]);
		F(3, 2) = (irm(2, 0) * d(1) - irm(1, 0) * d(2)) / (d[1] * d[1]);
		F(3, 3) = (irm(2, 2) * d(1) - irm(1, 2) * d(2)) / (d[1] * d[1]);*/

		// d pos / d dir
        F.template block<2, 2>(0, 2) = -p[1] * F.template block<2, 2>(2, 2);
		/*
		F(0, 2) = p[1] * (irm(1, 0) * d[0] - irm(0, 0) * d[1]) / (d[1] * d[1]);
		F(1, 2) = p[1] * (irm(1, 0) * d[2] - irm(2, 0) * d[1]) / (d[1] * d[1]);*/

		// d dir / d pos
        F.template block<2, 2>(2, 0) = Matrix2::Zero();
	}
	void predict(const Matrix<real, 3, 3> &invertedRotation, const Matrix<real, 3, 1> &translation, real posQ = 0.1, real dirQ = 0.1)
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
		P(0, 0) += posQ;
		P(1, 1) += posQ;
		P(2, 2) += dirQ;
		P(3, 3) += dirQ;
	}
	template<bool right>
	void correct(const CameraModel<real> &camera, const LineV &line, const Matrix2 &R)
	{
		// Calculate z and y
		Vector2 z(line.fa, line.fb), hx;
        camera.template projectLine<right>(hx, position, direction);
		Vector2 y = z - hx;

		// Calculate H
		Matrix<real, 2, 4> H;
        camera.template projectLineDiff<right>(H, position, direction);


		// Correct
		auto S = H * P * H.transpose() + R;
		auto K = P * H.transpose() * S.inverse();
		auto dx = K * y;
        position += dx.template block<2, 1>(0, 0);
        direction += dx.template block<2, 1>(2, 0);
		P = (Matrix4::Identity() - K * H) * P;
	}
	void correct(const CameraModel<real> &camera, const LineV &left, const LineV &right, const Matrix2 &R)
	{
		// Calculate z and y
		Vector2 lhx, rhx;
        camera.template projectLine<false>(lhx, position, direction);
        camera.template projectLine<true>(rhx, position, direction);
		Vector4 z(left.fa, left.fb, right.fa, right.fb);
		Vector4 hx; hx << lhx, rhx;
		Vector4 y = z - hx;

		// Calculate H
		Matrix<real, 2, 4> LH, RH;
        camera.template projectLineDiff<false>(LH, position, direction);
        camera.template projectLineDiff<true>(RH, position, direction);
		Matrix4 H;
        H.template block<2, 4>(0, 0) = LH;
        H.template block<2, 4>(2, 0) = RH;

		// Correct
		Matrix4 S = H * P * H.transpose();
        S.template block<2, 2>(0, 0) += R;
        S.template block<2, 2>(2, 2) += R;
		auto K = P * H.transpose() * S.inverse();
		auto dx = K * y;
        position += dx.template block<2, 1>(0, 0);
        direction += dx.template block<2, 1>(2, 0);
		P = (Matrix4::Identity() - K * H) * P;
	}
};
