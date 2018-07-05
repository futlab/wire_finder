#pragma once

template<typename real>
struct KalmanLine
{
	typedef Matrix<real, 3, 1> Vector3;
	typedef Matrix<real, 2, 1> Vector2;
	typedef Matrix<real, 5, 5> Matrix5;
	Vector3 direction;
	Vector2 position;
	Matrix5 P;

	KalmanLine(const CameraModel<real> &camera, const LineV & left, const LineV & right, uint value)
	{
		// Calculate position
		float leftX = left.b + (real)left.a / 32768.0f * camera.leftCenterY();
		float rightX = right.b + (real)right.a / 32768.0f * camera.rightCenterY();
		camera.pointByScreen(position, leftX, rightX);

		// Calculate direction
		camera.pointByScreen(direction, left.b, right.b, 0);
		direction[0] -= position[0];
		direction[2] -= position[1];
		direction.normalize();
	}
	KalmanLine(const Vector2 &position, const Vector3 &direction, real disp = 0) : position(position), direction(direction), P(Matrix5::Identity() * disp) {}
	static void getF(Matrix5 &F, const Matrix<real, 3, 3> &rotation, real dd0, real dd2)
	{
		F.block<3, 3>(2, 2) = rotation;
		F(0, 0) = rotation(0, 0) - rotation(1, 0) * dd0;
		F(0, 1) = rotation(0, 2) - rotation(1, 2) * dd0;
		F(1, 0) = rotation(2, 0) - rotation(1, 0) * dd2;
		F(1, 1) = rotation(2, 2) - rotation(1, 2) * dd2;
	}
	void predict(const Matrix<real, 3, 3> &rotation, const Matrix<real, 3, 1> &translation)
	{
		Vector3 p(position[0], 0, position[1]);
		p = rotation * p - translation;
		direction = rotation * direction;
		real dd0 = direction[0] / direction[1], dd2 = direction[2] / direction[1];
		position <<
			p[0] - p[1] * dd0,
			p[2] - p[1] * dd2;
		Matrix5 F = Matrix5::Zero();
		getF(F, rotation, dd0, dd2);
		P = F * P * F.transpose();
		// P += Q;
	}
	void correct(const CameraModel<real> &camera, const LineV & left, const LineV & right, uint value)
	{



	}

};
