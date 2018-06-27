#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <set>
#include "hough.h"
#include "cmplines.h"

#include "cl_utils0.h"

#include "linepool.h"

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
	cl::Set *set_;
    CLFilter(CLWrapper &cl, cv::Size size, cl::Set *set) :
        size_(size), cl_(&cl),
        clImage(&cl, size.width, size.height, nullptr, imageFormat<cl_uchar, 4>()/*, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY*/),
        clImageH(&cl, size.width, size.height, nullptr, imageFormat<cl_short, 4>(), CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY),
        clImageV(&cl, size.width, size.height, nullptr, imageFormat<cl_short, 4>(), CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY),
        clGradImage(&cl, size.width, size.height, nullptr, imageFormat<cl_short, 2>(), CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY),
        clGradMem(&cl, size.width * size.height * sizeof(cl_short) * 2),
        resH(size, CV_16SC4), resV(size, CV_16SC4), gradImage(size, CV_16SC2),
		hlv(set), set_(set)
    {
        //auto kernels = cl.loadKernels("kernel.cl", {"sobelRGBH", "sobelRGBV"});
        auto kernels = cl.loadKernels("kernel.cl", {"scharrRGBH", "scharrRGBV", "scharr5", "diffint"});
        diffH = kernels[0];
        diffV = kernels[1];
        diff5 = kernels[2];
        diffint = kernels[3];
		hlv.initialize(size, CV_8U, CV_16U, 150);
    }
	void printCounters()
	{
		std::cout << "Performance: " + hlv.getCounters() + "\n";
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
    cv::Mat process5(cv::Mat &src, const std::string &name, std::vector<LineV> &lines)
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
        //cv::imshow(name, diRes);

		cv::threshold(diRes, diRes, 2, 2, cv::THRESH_TRUNC);

		hlv.accumulateRows(diRes);
		//cv::imshow(name + " accRows", hlv.accRows_.readScaled());

		hlv.sumAccumulator();
		//std::vector<LineV> lines;
		{
			cv::Mat acc = hlv.accumulator.read();
			cv::Mat accRect = hlv.rectifyAccumulatorRef<ushort>(acc, size_.height);
            hlv.collectLinesRef<ushort, 4>(accRect, 350, lines, size_.height);

			double min, max;
			cv::minMaxLoc(accRect, &min, &max);
			accRect.convertTo(accRect, CV_8U, 255 / max);
			cv::imshow(name + " accRect", accRect);

			hlv.refineLines(lines);
			hlv.filterLines(lines);
			cv::imshow(name + " diRes", hlv.source_.readScaled());

            std::sort(lines.begin(), lines.end(), [](const auto & a, const auto & b) -> bool
            {
                return a.b > b.b;
            });
			return hlv.drawLines(src, lines);
		}

		hlv.collectLines();
		hlv.readLines(lines);
		cv::Mat acc = hlv.accumulator.readScaled();

		hlv.drawMarkers(acc, lines);
		cv::imshow(name + " accRows", acc);

		cv::Mat out = hlv.drawLines(src, lines);
		
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
    return cv::Mat((int)input.getHeight(), (int)input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}

int lcx = -1, lcy = -1;

void onMouseLC(int event, int x, int y, int, void*)
{
	if (event != cv::EVENT_LBUTTONDOWN)
		return;
	lcx = x;
	lcy = y;
}

inline int disparity(const LineV & left, const LineV & right)
{
    int
            lc = left.b + ((int(left.a) * 720) >> 15),
            rc = right.b + ((int(right.a) * 720) >> 15);

    return ((lc - rc) + (left.b - right.b)) / 2;
}

//void maximize std::vector<uint> &result, const std::vector<LineV> & left, const std::vector<LineV> & right

std::vector<pair<int, int>> findGoodPairs(std::vector<uint> &cmp, const std::vector<LineV> & left, const std::vector<LineV> & right)
{
    std::vector<pair<int, int>> good;
    int ls = left.size(), rs = right.size();
    assert(cmp.size() == ls * rs);
    if (cmp.empty()) return good;
    std::vector<bool> usedLeft, usedRight;
    usedLeft.resize(ls, false);
    usedRight.resize(rs, false);

    do {
        uint m = 600000;
        pair<int, int> p = std::make_pair(-1, -1);
        for (int l = 0; l < ls; ++l) if (!usedLeft[l])
            for (int r = 0; r < rs; ++r) if (!usedRight[r])
                if (uint v = cmp[r + l * rs]) {
                    if (v <= m) {
                        m = v;
                        p = std::make_pair(l, r);
                    }
                }
        if (p.first >= 0) {
            good.push_back(p);
            usedLeft[p.first] = true;
            usedRight[p.second] = true;
        } else break;
    } while(true);


    return good;
}

std::pair<int, int> findMaxDisparity(const std::vector<LineV> & left, const std::vector<LineV> & right, const std::vector<pair<int, int>> &good)
{
    std::pair<int, int> result = std::make_pair(-1, -1);
    uint m = 0;
    for (auto &g : good) {
        uint d = disparity(left[g.first], right[g.second]);
        if (d > m) {
            m = d;
            result = g;
        }
    }
    return result;
}

std::pair<int, int> drawLineCompare(std::vector<uint> &result, int left, int right, const std::vector<pair<int, int>> &good)
{
	if (!result.size()) return std::make_pair(-1, -1);
	assert(result.size() == left * right);
    auto copy = result;
    uint m = 0;
    for (auto &r : result) if (m < r) m = r;
    for (auto &r : result) if (r) r = m - r;

    cv::Mat out(cv::Size(right, left), CV_32S, result.data());
	double min, max;
	cv::minMaxLoc(out, &min, &max);
    cv::Mat res;
    out.convertTo(res, CV_8U, 255 / max);
    cv::cvtColor(res, res, CV_GRAY2RGB);
    for (auto &g : good) {
        res.at<cv::Vec3b>(g.first, g.second)[0] = 0;
    }

    cv::resize(res, res, cv::Size(400, 400), 0, 0, cv::INTER_NEAREST);
	std::pair<int, int> p = std::make_pair(-1, -1);
	if (lcx >= 0 && lcy >= 0) {
        cv::drawMarker(out, cv::Point(lcx, lcy), cv::Scalar(0), cv::MARKER_CROSS);
		p.first = lcy * left / 400;
		p.second = lcx * right / 400;
	}
    for (int l = 0; l < left; ++l)
        for (int r = 0; r < right; ++r)
            if(uint c = copy[r + l * right]) {
                cv::putText(res, std::to_string(c), cv::Point(400 * r / right + 5, 400 * l / left + 10), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0));
            }

	cv::imshow("Compare", res);
	cv::setMouseCallback("Compare", &onMouseLC, 0);
	return p;
}

void zedWork(CLFilter &f)
{
    sl::Camera zed;

    sl::InitParameters initParams;
    std::string svoFileName =
        //"d:/bag/svo/mrsk2.svo";
        "/home/igor/svo/mrsk/mrsk2.svo";
        //"/home/igor/svo/test2.svo";
    initParams.svo_input_filename = sl::String(svoFileName.c_str());
    initParams.sdk_verbose = true; // Enable the verbose mode
    initParams.depth_mode = sl::DEPTH_MODE_PERFORMANCE; // Set the depth mode to performance (fastest)
    initParams.coordinate_units = sl::UNIT_METER;
    initParams.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP;

    // Open the camera
    sl::ERROR_CODE err = zed.open(initParams);

    zed.enableTracking();
	LinePool pool;

    sl::Mat slLeft(zed.getResolution(), sl::MAT_TYPE_8U_C4, sl::MEM_CPU), slRight(zed.getResolution(), sl::MAT_TYPE_8U_C4, sl::MEM_CPU);
    cv::Mat left = slMat2cvMat(slLeft), right = slMat2cvMat(slRight);
    bool pause = false;
	std::vector<LineV> leftLines, rightLines;
	LinesCompare lc(f.set_);
	cl::MatBuffer leftMatBuf(f.set_, cv::Size(1280, 720), CV_8UC4), rightMatBuf(f.set_, cv::Size(1280, 720), CV_8UC4);
	lc.initialize(leftMatBuf, rightMatBuf);
    while(true) {
		sl::CalibrationParameters cp;
        if (!pause) {
            if (zed.grab(sl::RuntimeParameters(sl::SENSING_MODE_STANDARD, false, false)) == sl::SUCCESS) {
                zed.retrieveImage(slLeft, sl::VIEW_LEFT, sl::MEM_CPU);
                zed.retrieveImage(slRight, sl::VIEW_RIGHT, sl::MEM_CPU);
                cv::blur(left, left, cv::Size(3, 3));
                cv::blur(right, right, cv::Size(3, 3));

				auto ci = zed.getCameraInformation();
				cp = ci.calibration_parameters;
				pool.setCameraParams(cp.T.x, cp.left_cam.fx, cp.left_cam.cx, cp.left_cam.fy, cp.left_cam.cy, cp.right_cam.fx, cp.right_cam.cx, cp.right_cam.fy, cp.right_cam.cy);
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
		cv::Mat outL = f.process5(left, "left", leftLines);
        cv::Mat outR = f.process5(right, "right", rightLines);

		leftMatBuf.write(left);
		rightMatBuf.write(right);
		std::vector<uint> result;
		lc.stereoCompare(leftLines, rightLines, result);
		pool.onLines(leftLines, rightLines, result);
		pool.show(cp.left_cam.h_fov, cp.right_cam.h_fov);
        auto good = findGoodPairs(result, leftLines, rightLines);
        auto p = drawLineCompare(result, leftLines.size(), rightLines.size(), good);
        auto best = findMaxDisparity(leftLines, rightLines, good);
        if (best.first >= 0)
        {
            LineV& l = leftLines[best.first];
            cv::line(outL, cv::Point(l.b, 0), cv::Point(l.b + (((int(l.a) * outL.rows) >> 15)), outL.rows - 1), cv::Scalar(0, 200, 0), 3);
        }
        if (best.second >= 0)
        {
            LineV& l = rightLines[best.second];
            cv::line(outR, cv::Point(l.b, 0), cv::Point(l.b + (((int(l.a) * outR.rows) >> 15)), outR.rows - 1), cv::Scalar(0, 200, 0), 3);
        }
        if (p.first >= 0)
		{
			LineV& l = leftLines[p.first];
			cv::line(outL, cv::Point(l.b, 0), cv::Point(l.b + (((int(l.a) * outL.rows) >> 15)), outL.rows - 1), cv::Scalar(0, 0, 200), 3);
		}
		if (p.second >= 0)
		{
			LineV& l = rightLines[p.second];
			cv::line(outR, cv::Point(l.b, 0), cv::Point(l.b + (((int(l.a) * outR.rows) >> 15)), outR.rows - 1), cv::Scalar(0, 0, 200), 3);
		}

        //outL = leftMatBuf.read();
		//outR = rightMatBuf.read();

		cv::imshow("Left", outL);
        cv::imshow("Right", outR);

		f.printCounters();

        /*std::vector<cv::Mat> mv = {outL, outL/*cv::Mat(outL.size(), CV_8U, cv::Scalar(128))* /, outR};
        cv::Mat out;
        cv::merge(mv, out);
        cv::imshow("Stereo histogram", out);
        cv::setMouseCallback("Stereo histogram", &onMouse, 0 );*/

        //cv::imshow("Right result", outR);
        switch(cv::waitKey(1)) {
        case 'q': return;
		case 'p': pause = !pause; break;
		case '+': zed.setSVOPosition(zed.getSVOPosition() + 500);
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
		cl::Set set;// (cl.context_, cl.commandQueue_, { cl.deviceId_ });
		set.initializeDefault("CUDA");

        CLFilter f(cl, cv::Size(1280, 720), &set);
        //test(cl, k[0]);

        //stereoTest(cl);


        zedWork(f);
        //cv::Mat back = cv::imread("back.png");


        return 0;
    } catch (const cl::Error& e) {
        cout << "OpenCL error: " << e.what() << " code " << e.err()<< endl;
		return e.err();
    }
}
