#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include "hough.h"

#include "cl_utils0.h"

// ZED includes
#include <sl_zed/Camera.hpp>

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

class CLFilter
{
private:
    cv::Size size_;
    CLWrapper *cl_;
    CLImage2D clImage, clImageH, clImageV, clGradImage;
    CLMemory clGradMem;
    cv::Mat resH, resV, gradImage;
    cl_kernel diffH, diffV, diff5, diffint;
	HoughLinesV hlv;
public:
    CLFilter(CLWrapper &cl, cv::Size size, CLSet *set) :
        size_(size), cl_(&cl),
        clImage(&cl, size.width, size.height, nullptr, imageFormat<cl_uchar, 4>()/*, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY*/),
        clImageH(&cl, size.width, size.height, nullptr, imageFormat<cl_short, 4>(), CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY),
        clImageV(&cl, size.width, size.height, nullptr, imageFormat<cl_short, 4>(), CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY),
        clGradImage(&cl, size.width, size.height, nullptr, imageFormat<cl_short, 2>(), CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY),
        clGradMem(&cl, size.width * size.height * sizeof(cl_short) * 2),
        resH(size, CV_16SC4), resV(size, CV_16SC4), gradImage(size, CV_16SC2),
		hlv(set)
    {
        //auto kernels = cl.loadKernels("kernel.cl", {"sobelRGBH", "sobelRGBV"});
        auto kernels = cl.loadKernels("kernel.cl", {"scharrRGBH", "scharrRGBV", "scharr5", "diffint"});
        diffH = kernels[0];
        diffV = kernels[1];
        diff5 = kernels[2];
        diffint = kernels[3];
    }
    void showGrad(const cv::Mat &gradImage, const std::string &name)
    {
        cv::Mat buf, out;
        gradImage.convertTo(buf, CV_8UC2, 1.0 / 21, 128);
        std::vector<cv::Mat> mv;
        cv::split(buf, mv);
        //mv.insert(mv.begin(), mv[0]);
        //cv::merge(mv, out);
        cv::imshow(name, mv[0]);
    }
    void drawLines(cv::Mat m, const std::vector<cv::Vec4i> &lines, const cv::Scalar &color)
    {
        for( size_t i = 0; i < lines.size(); i++ )
        {
          cv::Vec4i l = lines[i];
          cv::line( m, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), color, 1, CV_AA);
        }
    }
    cv::Mat process5(const cv::Mat &src, const std::string &name)
    {
        cv::Mat buf;
        cv::resize(src, buf, size_);

        // calculate gradients
        cv::cvtColor(buf, buf, cv::COLOR_RGB2RGBA);
        clImage.write(buf.data);       
        clImage.setKernelArg(diff5, 0);
        clGradImage.setKernelArg(diff5, 1);
        cl_->exec(diff5, {(size_t)size_.width, (size_t)size_.height});
        clGradImage.read(gradImage.data);
        //showGrad(gradImage, name);

        // calculate diffint
        clGradMem.write(gradImage.data);
        clGradMem.setKernelArg(diffint, 0);
        CLMemory cldiResult(cl_, size_.width * size_.height);
        cldiResult.setKernelArg(diffint, 1);
        cl_uint w = size_.width, h = size_.height;
        setKernelArg(diffint, 2, w);
        setKernelArg(diffint, 3, h);
        cl_->exec(diffint, {(size_t) h});
        cv::Mat diRes(size_, CV_8U);
        cldiResult.read(diRes.data);
        cv::imshow(name, diRes);
		cv::Mat acc;
		hlv.accumulateRef<ushort>(diRes, acc);

		double min, max;
		cv::minMaxLoc(acc, &min, &max);
		acc.convertTo(acc, CV_8U, 255 / max);
		cv::imshow(name + " acc", acc);


        // HoughLines
        /*std::vector<cv::Mat> gvec;
        cv::split(gradImage, gvec);
        cv::Mat gHorHi, gHorLo;
        cv::threshold(gvec[0], gHorHi, 300, 255, cv::THRESH_BINARY);
        cv::threshold(gvec[0], gHorLo, -300, 255, cv::THRESH_BINARY);
        gHorLo.convertTo(gHorLo, CV_8U, -1, 255);
        gHorHi.convertTo(gHorHi, CV_8U);
        std::vector<cv::Mat> tv = {gHorHi * 0.3, cv::Mat::zeros(gradImage.size(), CV_8U), gHorLo * 0.3};
        cv::Mat test;


        cv::merge(tv, test);

        vector<cv::Vec4i> linesHi, linesLo;
        cv::HoughLinesP(gHorHi, linesHi, 1, CV_PI/180, 50, 400);
        drawLines(test, linesHi, cv::Scalar(255, 0, 0));
        cv::HoughLinesP(gHorLo, linesLo, 1, CV_PI/180, 50, 400);
        drawLines(test, linesLo, cv::Scalar(0, 0, 255));

        cv::imshow(name, test);*/


        //showGrad(gHorHi, name);

        cv::Mat hist(cv::Size(800, 600), CV_32F, cv::Scalar(0));
        const cl_short *pg = (cl_short *)gradImage.data;
        for (int y = 0; y < size_.height; y++)
            for (int x = 0; x < size_.width; x++, pg += 2) {
                int gv = pg[1], gh = pg[0];
                if (abs(gh) <= abs(gv)) continue;
                double phase = (double)gv / gh;
                double dx = x - (y - size_.height / 2) * phase;
                double offset = dx / size_.width * 2 - 1.0; //(dx * gh + dy * gv) / glen;
                int histX = int(400 + offset * 400);
                int histY = int(300 + phase * 300);
                if (histX < 0 || histY < 0 || histX >= hist.cols || histY >= hist.rows)
                    continue;
                float& pt = ((float *)hist.data)[histX + histY * hist.cols];
                if (gh > 0)
                    pt += gh * gh + gv * gv;
                else
                    pt -= gh * gh + gv * gv;
                /*uchar& pt = hist.data[histX + histY * hist.cols];
                if (gh > 0 && pt < 255) pt++;
                if (gh < 0 && pt > 0) pt--;
                if (gh > 0 && pt < 255) pt++;
                if (gh < 0 && pt > 0) pt--;
                if (gh > 0 && pt < 255) pt++;
                if (gh < 0 && pt > 0) pt--;*/
                //if (gh > 0 && pt < 255) pt++;
                //if (gh < 0 && pt > 0) pt--;
                //hist.data at<uchar>(histX, histY) = pt;
            }
        //double min, max;
        cv::minMaxLoc(hist, &min, &max);
        cv::Mat out;
        hist.convertTo(out, CV_8U, 128 / std::max(max, -min), 128);
        //for (unsigned char *c = hist.data; c < hist.dataend; ++c)
        //    if (*c == 128) *c = 0;

        //cv::imshow("Histogram " + name, hist);
        /*cv::Mat out;
        resH.convertTo(out, CV_8UC4, 1.0 / 32.0, 128);*/
        return out;
    }
};

float phase_ = NAN, offset_ = NAN;

static void onMouse(int event, int x, int y, int, void*)
{
    if( event != cv::EVENT_LBUTTONDOWN )
        return;
    offset_ = float(x - 400) / 400;
    phase_ = float(y - 300) / 300;
}

cv::Mat slMat2cvMat(sl::Mat& input)
{
    using namespace sl;
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}

void zedWork(CLFilter &f)
{
    sl::Camera zed;

    sl::InitParameters initParams;
    std::string svoFileName =
        "/home/igor/svo/mrsk/mrsk2.svo";
        //"/home/igor/svo/test2.svo";
    initParams.svo_input_filename = sl::String(svoFileName.c_str());
    initParams.sdk_verbose = true; // Enable the verbose mode
    initParams.depth_mode = sl::DEPTH_MODE_PERFORMANCE; // Set the depth mode to performance (fastest)

    // Open the camera
    sl::ERROR_CODE err = zed.open(initParams);

    sl::Mat slLeft(zed.getResolution(), sl::MAT_TYPE_8U_C4, sl::MEM_CPU), slRight(zed.getResolution(), sl::MAT_TYPE_8U_C4, sl::MEM_CPU);
    cv::Mat left = slMat2cvMat(slLeft), right = slMat2cvMat(slRight);
    bool pause = false;
    while(true) {
        if (!pause) {
            if (zed.grab(sl::RuntimeParameters(sl::SENSING_MODE_STANDARD, true, false)) == sl::SUCCESS) {
                zed.retrieveImage(slLeft, sl::VIEW_LEFT, sl::MEM_CPU);
                zed.retrieveImage(slRight, sl::VIEW_RIGHT, sl::MEM_CPU);
                cv::blur(left, left, cv::Size(3, 3));
                cv::blur(right, right, cv::Size(3, 3));
            }
        }

        cv::Mat left_, right_;
        left.copyTo(left_);
        right.copyTo(right_);
        if (!std::isnan(offset_) && !std::isnan(phase_)) {
            auto size = left.size();
            int x1 = (offset_ + 1) * size.width / 2 - phase_ * size.height / 2;
            int x2 = (offset_ + 1) * size.width / 2 + phase_ * size.height / 2;
            cv::line(left_, cv::Point(x1, 0), cv::Point(x2, size.height - 1), cv::Scalar(255, 0, 0));
            cv::line(right_, cv::Point(x1, 0), cv::Point(x2, size.height - 1), cv::Scalar(255, 0, 0));
        }
        cv::imshow("Left", left_);
        cv::imshow("Right", right_);
        cv::Mat outL = f.process5(left, "left");
        cv::Mat outR = f.process5(right, "right");

        std::vector<cv::Mat> mv = {outL, outL/*cv::Mat(outL.size(), CV_8U, cv::Scalar(128))*/, outR};
        cv::Mat out;
        cv::merge(mv, out);
        cv::imshow("Stereo histogram", out);
        cv::setMouseCallback("Stereo histogram", &onMouse, 0 );

        //cv::imshow("Right result", outR);
        switch(cv::waitKey(1)) {
        case 'q': return;
        case 'p': pause = !pause;
        }
    }
}

void stereoTest(CLWrapper &cl)
{
    cv::Size size(320, 240);
    cv::Mat left(size, CV_8U), right(size, CV_8U), result(size, CV_8U);
    uchar *l = left.data, *r = right.data;
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x, ++r, ++l)
        {
            *l = 2550 / (10 + abs(x - 160) + abs(y - 120));
            *r = 2550 / (10 + abs(x - 170) + abs(y - 120));
        }
    auto kernels = cl.loadKernels("kernel.cl", {"adCensus"});

    cl_kernel adCensus = kernels[0];
    CLMemory clLeft(&cl, left.data, left.total()), clRight(&cl, right.data, right.total()), clResult(&cl, result.total());
    clLeft.setKernelArg(adCensus, 0);
    clRight.setKernelArg(adCensus, 1);
    cl_uint step = size.width, width = size.width, height = size.height;
    setKernelArg(adCensus, 2, step);
    setKernelArg(adCensus, 3, width);
    setKernelArg(adCensus, 4, height);
    clResult.setKernelArg(adCensus, 5);
    cl.exec(adCensus, {(size_t)size.height / 10 * 32}, {32});
    clResult.read(result.data);


    cv::imshow("left", left);
    cv::imshow("right", right);
    cv::waitKey();
}

#include "hough.h"

int main()
{
    try {
        CLWrapper cl;
        cl.showDevices();
        std::string devName = cl.devInfoStr(CL_DEVICE_NAME);
        printf("Device name: %s\n", devName.c_str());
        cl.getImage2DFormats();
        //int ret = clEnqueueWriteBuffer()
        cl_bool imsup;
        cl.devInfo(CL_DEVICE_IMAGE_SUPPORT, &imsup);
        cl_ulong loc_size, glob_size;
        cl.devInfo(CL_DEVICE_LOCAL_MEM_SIZE, &loc_size);
        cl.devInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &glob_size);
        CLSet set(cl.context_, cl.commandQueue_, {cl.deviceId_});
        //houghTest(&set);

        CLFilter f(cl, cv::Size(1280, 720), &set);
        //test(cl, k[0]);

        //stereoTest(cl);


        zedWork(f);
        //cv::Mat back = cv::imread("back.png");


        return 0;
    } catch (const cl::Error& e) {
        cout << "OpenCL error: " << e.what() << " code " << e.err()<< endl;
    }
}
