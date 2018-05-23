
__kernel void test(__global int* message)
{
        // получаем текущий id.
        int gid = get_global_id(0);

        message[gid] += gid;
}

__kernel void sobel( __read_only image2d_t src, __write_only image2d_t dst)
{
	sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	uint4 v = read_imageui(src, srcSampler, pos);
	write_imageui(dst, pos, (uint4)(255, 0, 0, 0) - v);
}

__kernel void sobelRGBV(__read_only image2d_t src, __write_only image2d_t dst)
{
	sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	int x = get_global_id(0), y = get_global_id(1);
	int4 v = convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y - 1)));
	v +=    (convert_int4(read_imageui(src, srcSampler, (int2)(x,     y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x,     y - 1)))) << 1;
	v +=     convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y - 1)));

        write_imagei(dst, (int2)(x, y), v); // [-1020 : 1020]
}

__kernel void sobelRGBH(__read_only image2d_t src, __write_only image2d_t dst)
{
	sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	int x = get_global_id(0), y = get_global_id(1);
	int4 v = convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y - 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y - 1)));
	v +=    (convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y    ))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y    )))) << 1;
	v +=     convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y + 1)));

	write_imagei(dst, (int2)(x, y), v);
}

__kernel void scharrRGBV(__read_only image2d_t src, __write_only image2d_t dst)
{
	sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	int x = get_global_id(0), y = get_global_id(1);
        int4 v = (convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y - 1)))) * 3;
        v +=     (convert_int4(read_imageui(src, srcSampler, (int2)(x,     y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x,     y - 1)))) * 10;
        v +=     (convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y - 1)))) * 3;

        write_imagei(dst, (int2)(x, y), v); // [-4092 : 4092]
}

__kernel void scharrRGBH(__read_only image2d_t src, __write_only image2d_t dst)
{
	sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	int x = get_global_id(0), y = get_global_id(1);
        int4 v = (convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y - 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y - 1)))) * 3;
        v +=     (convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y    ))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y    )))) * 10;
        v +=     (convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y + 1)))) * 3;

	write_imagei(dst, (int2)(x, y), v);
}

__kernel void scharr5(__read_only image2d_t src, __write_only image2d_t dst)
{
        sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
        const int x = get_global_id(0), y = get_global_id(1);
        __const int4 k[3] = {(int4)(-3, -6, 6, 3), (int4)(-2, -2, 2, 2), (int4)(-1, -1, 1, 1)};

        int4 rv = (int4)(0, 0, 0, 0), rh = (int4)(0, 0, 0, 0);
        for(int dx = -2; dx <= 2; ++dx) {
            int lx = x + dx;
            int4 v = convert_int4((uint4)(read_imageui(src, sampler, (int2)(lx, y - 2)).x, read_imageui(src, sampler, (int2)(lx, y - 1)).x, read_imageui(src, sampler, (int2)(lx, y + 1)).x, read_imageui(src, sampler, (int2)(lx, y + 2)).x));
            rv = mad24(v, k[abs(dx)], rv);
            if (dx)
                rh = mad24(v, (int4)(1, 2, 2, 1), rh);
            else
                rh = -rh;
        }
        int4 vh = convert_int4((uint4)(read_imageui(src, sampler, (int2)(x - 2, y)).x, read_imageui(src, sampler, (int2)(x - 1, y)).x, read_imageui(src, sampler, (int2)(x + 1, y)).x, read_imageui(src, sampler, (int2)(x + 2, y)).x));
        rh = mad24(vh, k[0], rh);

        int4 r = (int4)(rh.lo + rh.hi, rv.lo + rv.hi);
        write_imagei(dst, (int2)(x, y), (int4)(r.even + r.odd, 0, 0)); // [-5355 : 5355]
}

#define CENSUSW 9
#define CENSUSH 7
int census(char *left, char *right, int step)
{
	uchar16 lref = (uchar16)(*left), rref = (uchar16)(*right);
	int d = mad24((CENSUSH >> 1), step, (CENSUSW >> 1));
	left -= d;
	right -= d;
	for (int y = CENSUSH; --y >= 0;) {
		
	
	}


}

__kernel void adCensus(__read_only image2d_t left, __read_only image2d_t right)
{


}

__kernel void gradHist(__read_only image2d_t h, __read_only image2d_t v, __write_only image2d_t hist)
{

}

/*__kernel void calcAxy(__read_only image2d_t gx, __read_only image2d_t gy, __global __write_only int *a)
{
	sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	int x = get_global_id(0), y = get_global_id(1);
	float4 am = (0, 0, 0, 0);
	for(int dx = 0; dx < 32; ++dx) {
		int4 h = read_imagei(gx, srcSampler, pos);
		int4 v = read_imagei(gy, srcSampler, pos);
		int2 sum = (int2)(h.x + h.y + h.z, v.x + v.y + v.z);
		am += (int4)(sum, sum) * (int4)(sum.odd, sum.even);
	}
}*/
