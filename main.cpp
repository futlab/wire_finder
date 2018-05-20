#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cl_utils.h"

using namespace std;

void test(CLWrapper &cl, cl_kernel &k)
{
	CLMemory buf(&cl, 40);
	int data[10] = {};
	buf.write(data);
	buf.setKernelArg(k, 0);
	cl.exec(k, {10});
	buf.read(data);
}

int main()
{
    CLWrapper cl;
	cl.getImage2DFormats();
    auto k = cl.loadKernels("kernel.cl", {"test", "sobelRGBH"});
	//int ret = clEnqueueWriteBuffer()
	cl_bool imsup;
	cl.devInfo(CL_DEVICE_IMAGE_SUPPORT, &imsup);
	cl_ulong loc_size, glob_size;
	cl.devInfo(CL_DEVICE_LOCAL_MEM_SIZE, &loc_size);
	cl.devInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &glob_size);
	std::string devName = cl.devInfoStr(CL_DEVICE_NAME);
	CLImage2D clImage(&cl, 717, 480, nullptr, imageFormat<cl_uchar, 4>());

	cv::Mat res(cv::Size(717, 480), CV_16SC4);
	CLImage2D clImageRes(&cl, 717, 480, res.data, imageFormat<cl_short, 4>(), CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_USE_HOST_PTR);

	test(cl, k[0]);

	cv::Mat back = cv::imread("back.png");
	cv::resize(back, back, cv::Size(717, 480));
	cv::cvtColor(back, back, cv::COLOR_RGB2RGBA);

	clImage.write(back.data);
	//clImage.read(back.data);
	//clImageRes.write(back.data);
	clImage.setKernelArg(k[1], 0);
	clImageRes.setKernelArg(k[1], 1);
	cl.exec(k[1], {717, 480});
	cl.finish();
	clImageRes.read(res.data);
	cv::Mat out;
	res.convertTo(out, CV_8UC4, 1.0 / 8.0, 128);

	cv::imshow("res", out);
	cv::waitKey();
    cout << "Hello World!" << endl;
    return 0;
}
