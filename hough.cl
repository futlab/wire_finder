#ifndef assert
#define assert(exp) (void)(0)
#endif

#ifndef WX
#define WX 320 // Width of group image window
#endif

#ifndef WY
#define WY 40  // Height of group image window
#endif

#ifndef ROW_TYPE
#define ROW_TYPE uchar
#endif

#ifndef ACC_TYPE
#define ACC_TYPE ushort
#endif

#ifndef ACC_H
#define ACC_H 128 // Height of accumulator, count of different angles
#endif

#ifndef ACC_W
#define ACC_W (WX + WY - 1) // Maximial width of the accumulator for image window
#endif

#ifndef WIDTH
#define WIDTH 1280
#endif

#ifndef HEIGHT
#define HEIGHT 720
#endif

#ifndef D
#define D 3
#endif

#ifndef MAX_LINES
#define MAX_LINES 1024
#endif

#define DF ((D) * 2 + 1)

#ifndef FULL_ACC_W
#define FULL_ACC_W (WIDTH + HEIGHT)
#endif

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

inline uint getShiftStepByA(uint yw)
{
    return (uint)(65536.f / (ACC_H - 1) * (2 * yw + WY - 1));
}

inline uint getShiftStepByYw(uint a)
{
	return (uint)(65536.f * (WY + (float)a / (ACC_H - 1) * (2 * WY)));
}

inline int getShift(uint xw, uint yw, uint a)
{
    int r = (int)(xw + yw + (float)a / (ACC_H - 1) * (2 * yw + WY - 1));
	assert(r == int((getShiftStart(xw, yw) + a * getShiftStep(yw)) >> 16));
	return r;
}

// Image size must be (WX * get_num_group(0); WY * get_num_group(1)); (1280; 720) = (160 * 8; 45 * 16)
__kernel /*__attribute__((reqd_work_group_size(32, 1, 1)))*/ void accumulateLocal(__global const uchar *src, uint step, __local ROW_TYPE *acc)
{
    const uint groupSize = get_local_size(0);

	// Initialize accumulator
	for (__local ROW_TYPE *pAcc = acc + get_local_id(0); pAcc < acc + ACC_H * ACC_W; pAcc += groupSize)
		*pAcc = 0;

	// Parameters of the group image window
	uint xw = WX * get_group_id(0), yw = WY * get_group_id(1), x = xw + get_local_id(0);

	// Shift parameters
	const uint 
		shiftStart = ((xw + yw) << 16) - yw * (65536 / ACC_H) + 0x8000,
		shiftStep = (2 * yw + WY - 1) * (65536 / ACC_H);

	// Scan image window
	__global const uchar *pSrc = src + mad24(yw, step, x);
	uint bStep = yw * (2 * 65536 / ACC_H);
	for (uint y = yw; y < yw + WY; y++) {
		uint bStart = (((x + y) << 16) | 0x8000) - y * (65536 / ACC_H);
		for (int xc = 0; xc < WX; xc += groupSize) {
			uchar value = pSrc[xc];
            __local ROW_TYPE *pAcc = acc;
			uint shift = shiftStart, b = bStart;
			for (uint a = 0; a < ACC_H; a++) {
				uint idx = (b - (shift & 0xFFFF0000)) >> 16;
                pAcc[idx] = add_sat(pAcc[idx], (ROW_TYPE)value);
				b -= bStep;
				shift -= shiftStep;
				pAcc += ACC_W;
			}
			bStart += groupSize << 16;
		}
		bStep += 2 * 65536 / ACC_H;
		pSrc += step;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

#define copy(d, s, l) for (int t__ = 0; t__ < (l); ++t__) (d)[t__] = (s)[t__]
#define delay(q, d, v) ((q)[0]); for (int t__ = 0; t__ < (d) - 1; ++t__) (q)[t__] = (q)[t__ + 1]; (q)[d - 1] = (v)


inline void copyRect(__global ROW_TYPE *dst, __local ROW_TYPE *src, uint dstStep, uint srcStep, uint width, uint height)
{
	uint gs = get_local_size(0), i = get_local_id(0);
	dst += i;
	src += i;
	for (uint y = height; y; --y) {
		for (uint x = 0; x < width; x += gs) {
			*dst = *src;
			dst += gs;
			src += gs;
		}
		dst += dstStep;
		src += srcStep;
	}
}

inline void addRect(__local ROW_TYPE *dst, __global ROW_TYPE *src, uint dstStep, uint srcStep, uint width, uint height)
{
	uint gs = get_local_size(0), i = get_local_id(0);
	dst += i;
	src += i;
	for (uint y = height; y; --y) {
		for (uint x = 0; x < width; x += gs) {
			*dst = add_sat(*dst, *src);
			dst += gs;
			src += gs;
		}
		dst += dstStep;
		src += srcStep;
	}
}

inline void zeroP(__local ACC_TYPE *dst, uint size)
{
	for (__local ACC_TYPE *d = dst + get_local_id(0), *e = dst + size; d < e; d += get_local_size(0))
		*d = 0;
}

inline void zeroPuint(__local uint *dst, uint size)
{
	for (__local uint *d = dst + get_local_id(0), *e = dst + size; d < e; d += get_local_size(0))
		*d = 0;
}


inline void setP(__local ACC_TYPE *dst, uint size, ACC_TYPE value)
{
	for (__local ACC_TYPE *d = dst + get_local_id(0), *e = dst + size; d < e; d += get_local_size(0))
		*d = value;
}

inline void copyP(global ACC_TYPE *dst, local ACC_TYPE *src, uint size)
{
	src += get_local_id(0);
	for (global ACC_TYPE *d = dst + get_local_id(0), *e = dst + size; d < e; d += get_local_size(0)) {
		*d = *src;
		src += get_local_size(0);
	}
}


__kernel /*__attribute__((reqd_work_group_size(32, 1, 1)))*/ void accumulate(__global const uchar *src, uint step, __global ROW_TYPE *dst)
{
	// Accumulate to local memory
	__local ROW_TYPE acc[ACC_H * ACC_W];
	accumulateLocal(src, step, acc);

	// Store accumulator to result
	const uint groupSize = get_local_size(0);
	__global ROW_TYPE *pDst = dst + (ACC_H * ACC_W) * mad24((uint)get_group_id(1), (uint)get_num_groups(0), (uint)get_group_id(0)) + get_local_id(0);
	const __local ROW_TYPE *pAcc = acc + get_local_id(0);
	for (uint i = ACC_W * ACC_H / groupSize; i; --i) {
		*pDst = *pAcc;
		pDst += groupSize;
		pAcc += groupSize;
	}
}

/*uint enter(__global volatile uint *flag) {
	local uint result;
	if (!get_local_id(0))
		result = !atomic_cmpxchg(flag, 0, 1);
	barrier(CLK_LOCAL_MEM_FENCE);
	return result;
}*/

__kernel /*__attribute__((reqd_work_group_size(32, 1, 1)))*/ void accumulateRows(__global const uchar *src, uint step, __global ROW_TYPE *dst, __global volatile uint *flags)
{
    local uint result;
    // Accumulate to local memory
	__local ROW_TYPE acc[ACC_H * ACC_W];
	accumulateLocal(src, step, acc);

	// Make full rows of accumulator
    uint flag_id = mad24((uint)get_group_id(1), (uint)get_num_groups(0) - 1, (uint)get_group_id(0));
    uint rowWidth = mad24((uint)get_num_groups(0), (uint)WX, (uint)(ACC_W - WX));
    dst += mad24(rowWidth, ACC_H * (uint)get_group_id(1), (uint)get_group_id(0) * WX);

    __global volatile uint *leftFlag = flags + flag_id - 1, *rightFlag = flags + flag_id;

    uchar works = 7, canWrite = 2, flagSet = 0; // 1: left, 2: center, 4: right
    if (get_group_id(0) == 0                    ) canWrite |= 1;
    else {
        if (!get_local_id(0))
            result = !atomic_cmpxchg(leftFlag, 0, 1);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (result)  { canWrite |= 1; flagSet |= 1; }
    }
    if (get_group_id(0) == get_num_groups(0) - 1) canWrite |= 4;
    else {
        if (!get_local_id(0))
            result = !atomic_cmpxchg(rightFlag, 0, 1);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (result) { canWrite |= 4; flagSet |= 4; }
    }

    /*if (!get_local_id(0))
        printf("Group %d (%d) - works: %d, canWrite: %d, flagSet: %d\n", get_group_id(0), get_local_id(0), works, canWrite, flagSet);
    uint ctr = 0;*/
    while (works /*&& ctr++ < 1000*/) {
        //if (!get_local_id(0))
        //	printf("Group %d (%d) - works: %d, canWrite: %d, flagSet: %d, ctr: %d\n", get_group_id(0), get_local_id(0), works, canWrite, flagSet, ctr);
        if ((works & 1) && !(canWrite & 1) && *leftFlag == 2) { // Sum and store left
            addRect(acc, dst,
                WX, rowWidth - (ACC_W - WX), ACC_W - WX, ACC_H);
            canWrite |= 1;
            barrier(CLK_LOCAL_MEM_FENCE);
            //if (!get_local_id(0)) printf("Group %d (%d) left add - works: %d, canWrite: %d, flagSet: %d\n", get_group_id(0), get_local_id(0), works, canWrite, flagSet);
        }
        if ((works & 4) && !(canWrite & 4) && *rightFlag == 2) { // Sum and store right
            addRect(acc + WX, dst + WX,
                WX, rowWidth - (ACC_W - WX), ACC_W - WX, ACC_H);
            canWrite |= 4;
            barrier(CLK_LOCAL_MEM_FENCE);
            //if (!get_local_id(0))	printf("Group %d (%d) right add - works: %d, canWrite: %d, flagSet: %d\n", get_group_id(0), get_local_id(0), works, canWrite, flagSet);
        }
        if (canWrite) {
            uint a = WX, b = ACC_W - WX;
            if (canWrite & 2) { a = ACC_W - WX; b = WX; }
            if (canWrite & 1) a = 0;
            if (canWrite & 4) b = ACC_W;

            uint w = b - a;
            copyRect( // Write result to global memory
                dst + a,				// dst
                acc + a,				// source
                rowWidth - w,			// dst step
                ACC_W -	w,				// src step
                w,						// width
                ACC_H);					// height

            barrier(CLK_GLOBAL_MEM_FENCE);

            uchar flagFin = flagSet & canWrite;
            if (!get_local_id(0)) {
                if (flagFin & 1) *leftFlag = 2;
                if (flagFin & 4) *rightFlag = 2;
            }
            flagSet &= ~flagFin;
            works &= ~canWrite;
            canWrite = 0;
            //if (!get_local_id(0)) printf("Group %d (%d) store - works: %d, canWrite: %d, flagSet: %d, a: %d, b: %d\n", get_group_id(0), get_local_id(0), works, canWrite, flagSet, a, b);
        }
    }
    // if (!get_local_id(0)) printf("Group %d (%d) exit - works: %d, canWrite: %d, flagSet: %d, ctr: %d\n", get_group_id(0), get_local_id(0), works, canWrite, flagSet, ctr);
}

void sumAngle(global const ROW_TYPE *accs, uint shiftF, uint shiftW, uint a, local ACC_TYPE *dst)
{
	zeroP(dst, FULL_ACC_W);
	const uint shiftWStep = (WY << 16) - WY * (1 + 2 * a) * (65536 / ACC_H);
	for (uint y = HEIGHT / WY; y; --y) 
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		local ACC_TYPE *d = dst + get_local_id(0) - (((int)shiftF) >> 16) + (((int)shiftW) >> 16);
		for (global const ROW_TYPE *ap = accs + get_local_id(0); ap < accs + WIDTH + (ACC_W - WX); ap += get_local_size(0)) {
			if ((uint)(d - dst) >= FULL_ACC_W)
				break;
			*d = add_sat(*d, (ACC_TYPE)*ap);
			//else printf("a: %d; d - dst = %d; shiftF: %d; shiftW: %d\n", a, (uint)(d - dst), shiftF, shiftW);
			d += get_local_size(0);
		}
		accs += (WIDTH + (ACC_W - WX)) * ACC_H;
		shiftW += shiftWStep;
	}
}

kernel void sumAccumulator(global const ROW_TYPE *accs, global ACC_TYPE *acc)
{
	__local ACC_TYPE temp[FULL_ACC_W];
	const uint s = ACC_H / get_num_groups(0);
	const uint i = s * get_group_id(0);
	acc += FULL_ACC_W * i;
	accs += (WIDTH + (ACC_W - WX)) * i;
	uint shiftFStep = (HEIGHT - 1) * (65536 / ACC_H);
	uint shiftF = 0x8000 - shiftFStep * i;
	uint shiftWStep = (WY - 1) * (65536 / ACC_H);
	uint shiftW = 0x8000 - shiftWStep * i;
	//if (i == 0 && get_local_id(0) == 0) printf("Shift full: %d(%d), shift window: %d(%d)\n", shiftF, shiftFStep, shiftW, shiftWStep);
	for (uint a = i; a < i + s; ++a) {
		//if (!get_local_id(0)) printf("a: %d, shiftF: %d, shiftW: %d\n", a, shiftF >> 16, shiftW >> 16);
		//if (!get_local_id(0) && get_group_id(0) == 2) printf("a: %d, shiftF: %d, shiftW: %d\n", a, shiftF >> 16, shiftW >> 16);
		sumAngle(accs, shiftF, shiftW, a, temp);
		barrier(CLK_LOCAL_MEM_FENCE);
		copyP(acc, temp, FULL_ACC_W);
		acc += FULL_ACC_W;
		accs += (WIDTH + (ACC_W - WX));
		shiftF -= shiftFStep;
		shiftW -= shiftWStep;
	}
}

inline ACC_TYPE rowMax(const ACC_TYPE *row)
{
	ACC_TYPE result = *(row++);
	for (const ACC_TYPE *e = row + 2 * D; row < e; ++row)
		result = max(result, *row);
	return result;
}

typedef struct __attribute__((packed)) _Line
{
	ushort value, width;
	short a, b;
	float fa, fb;
} Line;

// Global size: accumulator width; local size: any;
__kernel void collectLines(__global const ACC_TYPE *acc, uint threshold, uint step, __global volatile int *linesCount, __global Line *lines)
{
	__local Line localLines[MAX_LINES];
	__local volatile int localCounter;
	if (get_global_id(0) == 33)
		*linesCount = 0;
	if (!get_local_id(0))
		localCounter = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	ACC_TYPE row[DF], maxBuf[DF] = {}, valueQueue[D] = {};
	acc += get_global_id(0);
	uint d = 0;
	for (int y = 0; y < ACC_H; ++y) {
		// Load row to private memory
		copy(row, acc, DF);
		acc += step;

		// Find max and store to buf 
		maxBuf[d++] = rowMax(row);
		if (d >= DF) d = 0;

		// Get value from past
		ACC_TYPE value = delay(valueQueue, D, row[D]);

		/*if (get_global_id(0) == 73) {
			printf("y = %d; row = (%d, %d, %d, %d, %d); maxBuf = (%d, %d, %d, %d, %d); q = (%d, %d); value = %d\n", y, row[0], row[1], row[2], row[3], row[4], maxBuf[0], maxBuf[1], maxBuf[2], maxBuf[3], maxBuf[4], valueQueue[0], valueQueue[1], value);
		}*/
		// Add line if value is maximal in the area
		if (value > threshold && value >= rowMax(maxBuf)) {
			uint idx = atomic_inc(&localCounter);
			if (idx >= MAX_LINES) break;
			Line l = { value, 0, y - D, get_global_id(0) + D };
			localLines[idx] = l;
			//printf("Add line value %d at %d, %d\n", value, get_global_id(0), y - D);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	uint count = localCounter;
	// if (!get_local_id(0)) printf("Group %d, total line count %d, local %d\n", get_group_id(0), *linesCount, count);
	if (!count) return;
	__local uint lIdx;
	if (!get_local_id(0)) {
		lIdx = atomic_add(linesCount, count);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	uint idx = lIdx;

	if (idx >= MAX_LINES) return;
	__local Line *ll = localLines + get_local_id(0);
	if (idx + count > MAX_LINES)
		count = MAX_LINES - idx;
	for (__global Line *p = lines + idx + get_local_id(0), *e = lines + idx + count; p < e; p += get_local_size(0)) {
		*p = *ll;
		ll += get_local_size(0);
	}
}

__kernel void refineLines(__global const uchar *source, __global Line *lines)
{
	__global Line *line = lines + get_group_id(0);

	// Gathering statistical information
#define REFINE_STATS 16
#define REFINE_D 8
	/*__local volatile*/// LineStat stats[REFINE_STATS] = {};
	uint4 stats[REFINE_STATS] = {}; // sx, sxy, sy, sy2
	uint counters[REFINE_STATS] = {};
	const uint id = get_local_id(0);
	int xc = (line->b << 15) + id * line->a, step = mul24((int)line->a, (int)get_local_size(0));
	source += WIDTH * id;
	for (uint y = id; y < HEIGHT; y += get_local_size(0)) {
		int xt = xc >> 15;
		uint begin = max(xt - REFINE_D, (int)0);
		uint end = min(xt + REFINE_D, (int)(WIDTH - 1));
		// if (!get_local_id(0)) printf("begin: %d, end: %d\n", begin, end);
		for (int i = 0; i < (int)(end - begin); ++i) {
			uint x = i + begin;
			uint value = source[x] - 1;
			// source[x] += 1;
			if (value < REFINE_STATS) {
				stats[value] += (uint4)(x, mul24(x, y), y, mul24(y, y));
				counters[value]++;
			}
		}
		xc += step;
		source += WIDTH * get_local_size(0);
	}

	// Find maximal n
	__local volatile uint ns[REFINE_STATS];
	zeroPuint((local uint *)ns, REFINE_STATS);
	barrier(CLK_LOCAL_MEM_FENCE);
	for (uint x = 0; x < REFINE_STATS; ++x)
		atomic_add(ns + x, counters[x]);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint maxN = 0, maxX = 0;
	for (uint x = 0; x < REFINE_STATS; ++x) {
		uint n = ns[x];
		if (maxN < n) {
			maxN = n;
			maxX = x;
		}
	}
	
	// Sum koeffs for n and area around it
	uint4 stat = stats[maxX];
	if (maxX > 0) {
		stat += stats[maxX - 1]; maxN += counters[maxX - 1];
	}
	if (maxX < REFINE_STATS - 1) {
		stat += stats[maxX + 1]; maxN += counters[maxX + 1];
	}

	// Store results to local
	__local uint4 localStats[64];
	localStats[id] = stat;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (id) return;

	// Sum results in one item
	uint sx = 0, sy = 0;
	ulong sxy = 0, sy2 = 0;
	for (uint i = 0; i < get_local_size(0); i++) {
		uint4 s = localStats[i];
		sx += s.s0;
		sxy += s.s1;
		sy += s.s2;
		sy2 += s.s3;
	}

	//printf("n: %d, sx: %d, sxy: %d, sy: %d, sy2: %d\n", maxN, sx, (uint)sxy, sy, (uint)sy2);

	float d = (float)(sy2 * maxN - (long)sy * sy);
	if (fabs(d) < 1E-10) {
		printf("Too small d!\n");
		return;
	}
	float a = ((float)maxN * sxy - (float)sx * sy) / d;
	line->a = (short)(a * 32768);
	line->fa = a;

	float b = (sy2 * sx - sy * sxy) / d;
	line->b = (short)round(b);
	line->fb = b + a * (HEIGHT / 2);

	line->width = maxX;
}

