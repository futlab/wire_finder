
__kernel void test(__global int* message)
{
        // получаем текущий id.
        int gid = get_global_id(0);

        message[gid] += gid;
}

__kernel void sobel( __read_only image2d_t src, __write_only image2d_t dst)
{
	sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST ;
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	uint4 v = read_imageui(src, srcSampler, pos);
	write_imageui(dst, pos, (uint4)(255, 0, 0, 0) - v);
}

