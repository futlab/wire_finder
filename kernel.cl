
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

	write_imagei(dst, (int2)(x, y), v);
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

/*__kernel void scharrRGBV(__read_only image2d_t src, __write_only image2d_t dst)
{
	sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	int x = get_global_id(0), y = get_global_id(1);
	int4 v = convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y - 1)));
	v +=    (convert_int4(read_imageui(src, srcSampler, (int2)(x,     y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x,     y - 1)))) << 1;
	v +=     convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y - 1)));

	write_imagei(dst, (int2)(x, y), v);
}

__kernel void sobelRGBH(__read_only image2d_t src, __write_only image2d_t dst)
{
	sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	int x = get_global_id(0), y = get_global_id(1);
	int4 v = convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y - 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y - 1)));
	v +=    (convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y    ))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y    )))) << 1;
	v +=     convert_int4(read_imageui(src, srcSampler, (int2)(x + 1, y + 1))) - convert_int4(read_imageui(src, srcSampler, (int2)(x - 1, y + 1)));

	write_imagei(dst, (int2)(x, y), v);
}*/


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
