#include "MergeKernel.h"

#include <xpu/host.h>
#include <chrono>




class Timer{
public:
    Timer(){
        start = std::chrono::high_resolution_clock::now();
    }
    
    Timer(int size, int blocks, int threads){
        sze = size;
        blcks = blocks;
        thrds = threads;
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer(){
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        double time = dur*0.000000001;
        if(blcks == 0){
            printf("%f s -> %f ns \n", time, (double)dur);
        } else {
            //time in ns (double)dur
            printf("%i        %i        %i        %f        s\n", sze, blcks, thrds, time);
        }
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    int blcks = 0;
    int sze = 0;
    int thrds = 0;
};



int main() {

    static constexpr size_t N = 1000000;
    static constexpr int thrds = 512;

    xpu::initialize(xpu::driver::cuda);

    xpu::hd_buffer<float> a{N};
    xpu::hd_buffer<float> b{N};
    xpu::hd_buffer<float> dst{a.size() + b.size()};

    for (size_t i = 0; i < N; i++) {
        a.host()[i] = 2*i;
        b.host()[i] = 2*i+1;
    }

    xpu::copy(a, xpu::host_to_device);
    xpu::copy(b, xpu::host_to_device);
    for(int i = 1; i<=2048; i*=10)
    {
        Timer timer(N, i,thrds);
        xpu::run_kernel<GpuMerge>(xpu::grid::n_blocks(i), a.device(), a.size(), b.device(), b.size(), dst.device());
    }
    {
        Timer timer(N, 2000,thrds);
        xpu::run_kernel<GpuMerge>(xpu::grid::n_blocks(2000), a.device(), a.size(), b.device(), b.size(), dst.device());
    }

    xpu::copy(dst, xpu::device_to_host);

    float *h = dst.host();
    bool isSorted = true;
    for (size_t i = 1; i < dst.size(); i++) {
        isSorted &= (h[i-1] <= h[i]);
    }

    if (isSorted) {
        std::cout << "Data is sorted!" << std::endl;
    } else {
        for (size_t i = 0; i < dst.size(); i++) {
            std::cout << h[i] << " ";
            if (i % 10 == 9) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
        std::cout << "ERROR: Data is not sorted!" << std::endl;
    }

    return 0;
}
