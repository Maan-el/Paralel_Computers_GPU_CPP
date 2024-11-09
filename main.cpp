#include <chrono>
#include <iostream>
#include <random>
#include <omp.h>

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include <CL/cl.h>
#endif

static inline auto random_num(const float lbound, const float ubound) -> float;

static inline auto random_vec(const size_t n) -> std::vector<float>;

static inline auto roundup(int a, int b) -> int;

static inline auto divup(int a, int b) -> int;

static inline void check(const cl_int err, const std::string_view context);

static inline void check_build(const cl_program program, const cl_device_id device, const cl_int err);

auto step(std::span<float> r, const std::span<const float> d, const size_t n) -> void;

#define CHECK(x) check(x, #x);

const char *kernel_source =
        R""(
__kernel void my_padded_kernel(__global const float* r, __global float* d, int n, int nn) {
    int ja = get_local_id(0);
    int i = get_group_id(1);

    __global float* t = d + nn * nn;

    for (int jb = 0; jb < nn; jb += 64) {
        int j = jb + ja;
        float v = (i < n && j < n) ? r[n*i + j] : HUGE_VALF;
        d[nn*i + j] = v;
        t[nn*j + i] = v;
    }
}

__kernel void my_kernel(__global float* r, __global const float* d, int n, int nn) {
    int ia = get_local_id(0);
    int ja = get_local_id(1);
    int ic = get_group_id(0);
    int jc = get_group_id(1);

    __global const float* t = d + nn * nn;

    float v[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            v[ib][jb] = HUGE_VALF;
        }
    }
    for (int k = 0; k < n; ++k) {
        float x[8];
        float y[8];
        for (int ib = 0; ib < 8; ++ib) {
            int i = ic * 64 + ib * 8 + ia;
            x[ib] = t[nn*k + i];
        }
        for (int jb = 0; jb < 8; ++jb) {
            int j = jc * 64 + jb * 8 + ja;
            y[jb] = d[nn*k + j];
        }
        for (int ib = 0; ib < 8; ++ib) {
            for (int jb = 0; jb < 8; ++jb) {
                v[ib][jb] = min(v[ib][jb], x[ib] + y[jb]);
            }
        }
    }
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < n && j < n) {
                r[n*i + j] = v[ib][jb];
            }
        }
    }
}
)"";


auto main() -> int {
    constexpr auto n = 6300;

    const std::vector<float> read = random_vec(n);
    std::vector<float> write(read.size());

    auto t1 = std::chrono::high_resolution_clock::now();
    step(write, read, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto exec_time = std::chrono::duration<double, std::milli>(t2 - t1);

    std::cout << "Function exec time: " << exec_time.count() << "ms" << std::endl;
    return 0;
}

auto step(std::span<float> r, const std::span<const float> d, const size_t n) -> void {
    // Sanity Check
    if (r.size() != d.size()) { return; }

    int nn = roundup(n, 64);

    // Setup
    cl_int err;
    cl_platform_id platform;
    CHECK(clGetPlatformIDs(1, &platform, nullptr));

    cl_device_id device;
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    check(err, "clCreateContext");

#ifdef CL_VERSION_2_0
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    check(err, "clCreateCommandQueue");

    // Compile Kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, &err);
    check(err, "clCreateProgramWithSource");
    check_build(program, device, clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));

    cl_kernel padded_kernel = clCreateKernel(program, "my_padded_kernel", &err);
    check(err, "clCreateKernel");
    cl_kernel kernel = clCreateKernel(program, "my_kernel", &err);
    check(err, "clCreateKernel");

    // Allocate memory & copy data to the cpu
    cl_mem dGPU = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * nn * nn * sizeof(float), nullptr, &err);
    check(err, "clCreateBuffer");

    cl_mem rGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * n * sizeof(float),
                                 (void *) d.data(), &err);
    check(err, "clCreateBuffer");

    // Run kernel
    {
        size_t wlsize[2] = {64, 1};
        size_t wgsize[2] = {64, size_t(nn)};
        CHECK(clSetKernelArg(padded_kernel, 0, sizeof(cl_mem), &rGPU));
        CHECK(clSetKernelArg(padded_kernel, 1, sizeof(cl_mem), &dGPU));
        CHECK(clSetKernelArg(padded_kernel, 2, sizeof(int), &n));
        CHECK(clSetKernelArg(padded_kernel, 3, sizeof(int), &nn));
        CHECK(clEnqueueNDRangeKernel(
            queue, padded_kernel, 2, nullptr, wgsize,
            wlsize, 0, nullptr, nullptr
        ));
        CHECK(clFinish(queue));
    }

    // Run kernel
    {
        size_t wlsize[2] = {8, 8};
        size_t wgsize[2] = {size_t(nn / 8), size_t(nn / 8)};
        CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &rGPU));
        CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dGPU));
        CHECK(clSetKernelArg(kernel, 2, sizeof(int), &n));
        CHECK(clSetKernelArg(kernel, 3, sizeof(int), &nn));
        CHECK(clEnqueueNDRangeKernel(
            queue, kernel, 2, nullptr, wgsize, wlsize, 0, nullptr, nullptr
        ));
        CHECK(clFinish(queue));
    }

    // Copy data back to GPU & release memory
    CHECK(clEnqueueReadBuffer(queue,rGPU, true, 0, n * n * sizeof(float), r.data(), 0, nullptr, nullptr));
    CHECK(clReleaseMemObject(rGPU));
    CHECK(clReleaseMemObject(dGPU));

    // Release everything else
    CHECK(clReleaseKernel(kernel));
    CHECK(clReleaseProgram(program));
    CHECK(clReleaseCommandQueue(queue));
    CHECK(clReleaseContext(context));
}

static inline auto divup(int a, int b) -> int {
    return (a + b - 1) / b;
}

static inline auto roundup(int a, int b) -> int {
    return divup(a, b) * b;
}

static inline void check(const cl_int err, const std::string_view context) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error " << err << ": " << context << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline void check_build(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        std::cerr << "OpenCL build failure: " << std::endl;
        size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, ' ');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        std::cout << log << std::endl;
        std::exit(EXIT_FAILURE);
    } else if (err != CL_SUCCESS) {
        std::cerr << "OpenCL build failed: " << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline auto random_vec(const size_t n) -> std::vector<float> {
    std::vector<float> buf(n * n);

    static std::vector<std::mt19937> generators;

#pragma omp parallel
    {
#pragma omp single
        {
            if (generators.begin() == generators.end()) {
                for (int i = 0; i < omp_get_num_threads(); ++i) {
                    generators.emplace_back(std::random_device{}());
                }
            }
        }

        std::mt19937 &g = generators[omp_get_thread_num()];
        std::uniform_real_distribution<float> distrib{0.f, 10.f};

#pragma omp for
        for (size_t i = 0; i < buf.size(); ++i) {
            buf[i] = distrib(g);
        }
    }

    return buf;
}
