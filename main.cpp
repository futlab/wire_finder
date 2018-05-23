#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>

#include "cl_utils.h"

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
    cv::Mat resH, resV, gradImage;
    cl_kernel diffH, diffV, diff5;
public:
    CLFilter(CLWrapper &cl, cv::Size size) :
        size_(size), cl_(&cl),
        clImage(&cl, size.width, size.height, nullptr, imageFormat<cl_uchar, 4>()/*, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY*/),
        clImageH(&cl, size.width, size.height, nullptr, imageFormat<cl_short, 4>(), CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY),
        clImageV(&cl, size.width, size.height, nullptr, imageFormat<cl_short, 4>(), CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY),
        clGradImage(&cl, size.width, size.height, nullptr, imageFormat<cl_short, 2>(), CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY),
        resH(size, CV_16SC4), resV(size, CV_16SC4), gradImage(size, CV_16SC2)
    {
        //auto kernels = cl.loadKernels("kernel.cl", {"sobelRGBH", "sobelRGBV"});
        auto kernels = cl.loadKernels("kernel.cl", {"scharrRGBH", "scharrRGBV", "scharr5"});
        diffH = kernels[0];
        diffV = kernels[1];
        diff5 = kernels[2];
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
    cv::Mat process5(const cv::Mat &src, const std::string &name)
    {
        cv::Mat buf;
        cv::resize(src, buf, size_);
        cv::cvtColor(buf, buf, cv::COLOR_RGB2RGBA);
        clImage.write(buf.data);
        //clImage.read(back.data);
        //clImageRes.write(back.data);

        clImage.setKernelArg(diff5, 0);
        clGradImage.setKernelArg(diff5, 1);
        cl_->exec(diff5, {(size_t)size_.width, (size_t)size_.height});
        clGradImage.read(gradImage.data);
        //showGrad(gradImage, name);

        cv::Mat hist(cv::Size(800, 600), CV_8U, cv::Scalar(128));
        const cl_short *pg = (cl_short *)gradImage.data;
        for (int y = 0; y < size_.height; y++)
            for (int x = 0; x < size_.width; x++, pg += 2) {
                int gv = pg[1], gh = pg[0];
                if (abs(gh) < 20) continue;
                double glen = sqrt(gv * gv + gh * gh);
                double phase = gv / glen;
                int dx = x - size_.width / 2, dy = y - size_.height / 2;
                double offset = (dx * gh + dy * gv) / glen;
                if (gh < 0) {
                    offset = -offset;
                    phase = - phase;
                }
                int histX = int(400 + offset * 0.5);
                int histY = int(300 + phase * 290);
                if (histX < 0 || histY < 0 || histX >= hist.cols || histY >= hist.rows)
                    continue;
                uchar& pt = hist.data[histX + histY * hist.cols];
                if (gh > 0 && pt < 255) pt++;
                if (gh < 0 && pt > 0) pt--;
                if (gh > 0 && pt < 255) pt++;
                if (gh < 0 && pt > 0) pt--;
                //if (gh > 0 && pt < 255) pt++;
                //if (gh < 0 && pt > 0) pt--;
                //hist.data at<uchar>(histX, histY) = pt;
            }

        //cv::imshow("Histogram " + name, hist);
        /*cv::Mat out;
        resH.convertTo(out, CV_8UC4, 1.0 / 32.0, 128);*/
        return hist;
    }

    cv::Mat process(const cv::Mat &src, const std::string &name)
    {
        cv::Mat buf;
        cv::resize(src, buf, size_);
        cv::cvtColor(buf, buf, cv::COLOR_RGB2RGBA);
        clImage.write(buf.data);
        //clImage.read(back.data);
        //clImageRes.write(back.data);

        cl_kernel k = diffH;
        clImage.setKernelArg(k, 0);
        clImageH.setKernelArg(k, 1);
        cl_->exec(k, { (size_t)size_.width, (size_t)size_.height});
        clImageH.read(resH.data);

        k = diffV;
        clImage.setKernelArg(k, 0);
        clImageV.setKernelArg(k, 1);
        cl_->exec(k, {(size_t)size_.width, (size_t)size_.height});
        clImageV.read(resV.data);

        cv::Mat hist(cv::Size(800, 600), CV_8U, cv::Scalar(128));
        const cl_short *pv = (cl_short *)resV.data, *ph = (cl_short *)resH.data;
        for (int y = 0; y < size_.height; y++)
            for (int x = 0; x < size_.width; x++, pv += 4, ph += 4) {
                int gv = pv[0] + pv[1] + pv[2], gh = ph[0] + ph[1] + ph[2];
                if (abs(gh) < 20) continue;
                double glen = sqrt(gv * gv + gh * gh);
                double phase = gv / glen;
                int dx = x - size_.width / 2, dy = y - size_.height / 2;
                double offset = (dx * gh + dy * gv) / glen;
                if (gh < 0) {
                    offset = -offset;
                    phase = - phase;
                }
                int histX = int(400 + offset * 0.5);
                int histY = int(300 + phase * 290);
                if (histX < 0 || histY < 0 || histX >= hist.cols || histY >= hist.rows)
                    continue;
                uchar& pt = hist.data[histX + histY * hist.cols];
                if (gh > 0 && pt < 255) pt++;
                if (gh < 0 && pt > 0) pt--;
                if (gh > 0 && pt < 255) pt++;
                if (gh < 0 && pt > 0) pt--;
                //if (gh > 0 && pt < 255) pt++;
                //if (gh < 0 && pt > 0) pt--;
                //hist.data at<uchar>(histX, histY) = pt;
            }

        //cv::imshow("Histogram " + name, hist);
        /*cv::Mat out;
        resH.convertTo(out, CV_8UC4, 1.0 / 32.0, 128);*/
        return hist;
    }
};

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
    std::string svoFileName = "/home/igor/svo/test2.svo";
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

        cv::imshow("Left", left);
        cv::imshow("Right", right);
        cv::Mat outL = f.process5(left, "left");
        cv::Mat outR = f.process5(right, "right");

        std::vector<cv::Mat> mv = {outL, outL/*cv::Mat(outL.size(), CV_8U, cv::Scalar(128))*/, outR};
        cv::Mat out;
        cv::merge(mv, out);
        cv::imshow("Stereo histogram", out);
        //cv::imshow("Right result", outR);
        switch(cv::waitKey(1)) {
        case 'q': return;
        case 'p': pause = !pause;
        }
    }
}

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

        CLFilter f(cl, cv::Size(1280, 720));
        //test(cl, k[0]);
        zedWork(f);
        //cv::Mat back = cv::imread("back.png");


        return 0;
    } catch (const CLError& e) {
        cout << "OpenCL error: " << e.what() << endl;
    }
}
