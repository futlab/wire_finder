#include "cl_utils.h"
#include <iostream>

int main()
{
    using namespace std;
    try {
        cl::printCLDevices();
        string code = "__kernel void sum(__global const uchar *a, __global const uchar *b, __global uchar *r) { uint x = get_global_id(0); r[x] = a[x] + b[x]; } ";

        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform &platform = platforms[0];
        vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto &device : devices) {
            cout << "Testing device " << device.getInfo<CL_DEVICE_NAME>() << ":" << endl;
            cl::Context ctx(device);
            cl::CommandQueue cq(ctx);
            cl::Program program(ctx, code);
            try {
                program.build({device});
            } catch(cl::BuildError &e) {
                cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
                string name     = device.getInfo<CL_DEVICE_NAME>();
                string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                cerr << "Build log for code on device " << name << ":" << endl
                        << buildlog << endl;
            }
            cl::Kernel kernel(program, "sum");
            cout << "Preferred work group size multiple: " << kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << endl;

            const size_t size = 10240;
            cl_uchar a[size], b[size], r[size] = {};
            for (size_t x = 0; x < size; x++) {
                a[x] = cl_uchar(x);
                b[x] = cl_uchar(200 - 2 * x);
            }

            cl::Buffer aBuf(ctx, CL_MEM_READ_WRITE, size);
            cl::Buffer bBuf(ctx, CL_MEM_READ_WRITE, size);
            cl::Buffer rBuf(ctx, CL_MEM_READ_WRITE, size);
            cq.enqueueWriteBuffer(aBuf, true, 0, size, a);
            cq.enqueueWriteBuffer(bBuf, true, 0, size, b);
            kernel.setArg(0, aBuf);
            kernel.setArg(1, bBuf);
            kernel.setArg(2, rBuf);
            cq.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(size));
            cq.enqueueReadBuffer(rBuf, true, 0, size, r);
            for (size_t x = 0; x < size; x++)
                if (cl_uchar(a[x] + b[x]) != r[x])
                {
                    cout << "Wrong result: " << a[x] << " + " << b[x] << " != " << r[x] << " (x = " << x << endl;
                    return 1;
                }
            cout << "Test completed" << endl;
        }
        return 0;
    } catch (const cl::Error& e) {
        cout << "OpenCL error: " << e.what() << " code " << e.err()<< endl;
        return 2;
    }
}
