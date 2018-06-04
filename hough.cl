#define WX 640 // Width of group image window
#define WY 32  // Height of group image window

#ifndef ACC_TYPE
#define ACC_TYPE uchar
#endif

#define ACC_H 128 // Height of accumulator, count of different angles
#define ACC_W (WX + WY - 1) // Maximial width of the accumulator
#define MAX_WIDTH 1280
#define ACC_W_FULL (1920 + 1080)

#ifndef GS
#define GS 32
#endif

// Line equation: x = b + y * a; a = tg(angle); -1.0 <= a <= 1.0, bacause we need only vertical lines.
// Group image window on scan: xw <= x < xw + WX; yw <= y < yw + WY,
//		where (xw, yw) - left-upper corner of the group window
// Image to accumulator transform: b = x - y * a, where a in [-1.0; 1.0]
// Accumulator b range if a ==  1.0: [x - y] = [xw - yw - WY + 1; xw + WX - 1 - yw]
// Accumulator b range if a == -1.0: [x + y] = [xw + yw; xw + WX + yw + WY - 2]
// Maximal accumulator b range width: WX + WY - 2
// Full image b range: [x - y * a] = a >= 0 : [-yMax * a; xMax]; a < 0 : [0; xMax - yMax *a) ]
// Full image b range width: xMax + yMax

inline uint getShiftStart(uint xw, uint yw) 
{ 
	return (xw + yw) << 16;
}

inline uint getShiftStep(uint yw)
{
	return uint(65536.f / (ACC_H - 1) * (2 * yw + WY - 1));
}

inline int getShift(uint xw, uint yw, uint a)
{
	int r = int(xw + yw + (float)a / (ACC_H - 1) * (2 * yw + WY - 1));
	assert(r == int((getShiftStart(xw, yw) + a * getShiftStep(yw)) >> 16));
	return r;
}

// Image size must be (WX * get_num_group(0); WY * get_num_group(1)); (1280; 720) = (640 * 2; 32 * 22.5?)
__kernel /*__attribute__((reqd_work_group_size(32, 1, 1)))*/ void houghScan(__global const uchar *src, uint step, __global ACC_TYPE *dst)
{
	const uint groupSize = get_group_size(0);

	// Declare and initialize accumulator
	__local ACC_TYPE acc[ACC_H * ACC_W];
	for (__local ACC_TYPE *pAcc = acc + get_local_id(0); pAcc < acc + ACC_H * ACC_W; pAcc += groupSize)
		*pAcc = 0;

	// Parameters of the group image window
	uint xw = WX * get_group_id(0), yw = WY * get_group_id(1), x = xw + get_local_id(0);

	// Shift parameters
	const uint shiftStart = getShiftStart(xw, yw), shiftStep = getShiftStep(yw);

	// Scan image window
	__global const uchar *pSrc = src + mad24(yw, step, x);
	for (uint y = yw; y < yw + WY; y++) {
		uint bStep = uint(2.0f / (ACC_H - 1) * 65536.f * y);
		for (int xc = 0; xc < WX; xc += groupSize) {
			uchar value = pSrc[xc];
			__local ACC_TYPE *pAcc = acc;
			uint b = ((x + xc + y) << 16) | 0x8000;
			uint shift = shiftStart;
			for (uint a = 0; a < ACC_H; a++) {
				float bf = x + xc - y * ((float)a * 2.0f / (ACC_H - 1) - 1.0f);
				assert(round(bf) == (signed(b) >> 16));
				uint idx = (b - shift & 0xFFFF0000) >> 16;
				assert(idx < ACC_W);
				pAcc[idx] = add_sat(pAcc[idx], value);
				b -= bStep;
				shift -= shiftStep;
				pAcc += ACC_W;
			}
		}
		pSrc += step;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store accumulator to result
	__global ACC_TYPE *pDst = dst + (ACC_H * ACC_W) * mad24(get_group_id(1), get_num_groups(0), get_group_id(0));
	const __local ACC_TYPE *pAcc = acc;
	for (int i = ACC_W * ACC_H / groupSize; i; --i) {
		*pDst = *pAcc;
		pDst += groupSize;
		pAcc += groupSize;
	}
}

/*typedef struct __attribute__((packed)) _Line
{
	uchar weight, a;
	short b;
} Line;*/

// get_num_groups(0) must be equal to ACC_H
__kernel void houghGrab(__global const ACC_TYPE *accs, uint width, uint height, /*__global Line *lines*/ __global ACC_TYPE *res)
{
	const uint groupSize = get_group_size(0);

	// Declare and initialize accumulator
	ushort fullAcc[ACC_W_FULL / GS] = { 0 };
	uchar counters[ACC_W_FULL / GS] = { 0 }; // Count of rows, where acc > threshold
	//for (__local ACC_TYPE *paa = angleAcc + get_local_id(0); paa < angleAcc + ACC_W_FULL; paa += groupSize)
	//	*paa = 0;

	uint a = get_group_id(0);
	__global const ACC_TYPE *pAcc = accs + a * ACC_W;

	for (uint yw = 0; yw < height; yw += WY) {
		ACC_TYPE rowAcc[(MAX_WIDTH + 2 * (WY - 1)) / GS] = { 0 };
		for (uint xw = 0; xw < width; xw += WX) {
			uint shift = getShift(xw, yw, a);
			for (uint x = xw + get_local_id(0); x < xw + WX; x += groupSize) {
				ACC_TYPE value = pAcc[x];
				uint idx = WY - 1 - shift;
				rowAcc[idx] = add_sat(rowAcc[idx], value);
			}
			pAcc += ACC_H * ACC_W;
		}
		memcpy(res, rowAcc, sizeof(rowAcc));
	}
}
