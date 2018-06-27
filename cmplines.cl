
#ifndef WIDTH
#define WIDTH 1280
#endif

#ifndef HEIGHT
#define HEIGHT 720
#endif

#ifndef CMP_D
#define CMP_D 8
#endif

#ifndef MIN_DISP
#define MIN_DISP 5
#endif

#ifndef MAX_DISP
#define MAX_DISP (WIDTH / 2)
#endif

#ifndef MAX_DA
#define MAX_DA (32768 / 4)
#endif

typedef struct __attribute__((packed)) _Line
{
	ushort value, width;
	short a, b; // x = ((a * y) >> 16) + b;
} Line;

void load(local uchar4 *buf, global uchar4 *source, int a, int b)
{
	const uint id = get_local_id(0);
	int xc = (b << 15) + id * a, step = mul24(a, (int)get_local_size(0));
	source += WIDTH * id;
	buf += 2 * CMP_D * id;
	for (uint y = id; y < HEIGHT; y += get_local_size(0)) {
		uint x = (xc >> 15) - CMP_D;
		for (int i = 0; i < 2 * CMP_D; ++i, ++x)
		{
			buf[i] = (x < WIDTH) ? source[x] : (uchar4)(0);
			// if (x < WIDTH) source[x] = (uchar4)(0);
		}
		xc += step;
		source += WIDTH * get_local_size(0);
		buf += 2 * CMP_D * get_local_size(0);
	}
}

uint compare(local uchar4 *buf, global uchar4 *image, int a, int b)
{
	local volatile uint result;
	uint r = 0;
	const uint id = get_local_id(0);
	if (!id) result = 0;
	int xc = (b << 15) + id * a, step = mul24(a, (int)get_local_size(0));
	image += WIDTH * id;
	buf += 2 * CMP_D * id;
	for (uint y = id; y < HEIGHT; y += get_local_size(0)) {
		uint x = (xc >> 15) - CMP_D;
		for (int i = 0; i < 2 * CMP_D; ++i, ++x) if (x < WIDTH)
		{
			uchar4 v = abs_diff(buf[i], image[x]);
			r += v.x + v.y + v.z;
		}
		xc += step;
		image += WIDTH * get_local_size(0);
		buf += 2 * CMP_D * get_local_size(0);
	}
	atomic_add(&result, r);
	barrier(CLK_LOCAL_MEM_FENCE);
	return result;
}

// Pixel format: RGBA, where A is mask
kernel void compareLinesStereo(global uchar4 *left, global uchar4 *right, const global Line *leftLines, const global Line *rightLines, uint rightCount, global uint *result)
{
	local uchar4 localLine[HEIGHT * CMP_D * 2];
	const uint id = get_group_id(0);
	const global Line *leftLine = leftLines + id, *rightLine = rightLines;
	int 
		leftA = leftLine->a, 
		leftB = leftLine->b,
		leftC = leftB + (((int)leftA * HEIGHT) >> 15);

	load(localLine, left, leftA, leftB);
	barrier(CLK_LOCAL_MEM_FENCE);
	result += id * rightCount;
	for (uint rc = 0; rc < rightCount; ++rc, ++rightLine) {
		uint r = 0;
		int 
			rightA = rightLine->a, 
			rightB = rightLine->b,
			rightC = rightB + (((int)rightA * HEIGHT) >> 15);

		int
			dB = leftB - rightB,
			dC = leftC - rightC;

		if (abs_diff(leftA, rightA) < MAX_DA && dB >= MIN_DISP && dB <= MAX_DISP && dC >= MIN_DISP && dC <= MAX_DISP)
			r = compare(localLine, right, rightA, rightB);

		*(result++) = r;
	}
}
