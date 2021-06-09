#include "MergeKernel.h"

#include <xpu/host.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


class Timer{
public:
    Timer(){
        start = std::chrono::high_resolution_clock::now();
    }
    
    Timer(int size, int blocks, int blocksize){
        sze = size;
        blcks = blocks;
        blcksze = blocksize;
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer(){
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        long double time = (long double)dur*(long double)0.000000001;
        if(blcks == 0){
            printf("%Lf s -> %Lf ns \n", time, (long double)dur);
        } else {
            //time in ns (double)dur
            printf("%i        %Lfs        %i        %i\n", blcksze, time, sze, blcks);
            std::ofstream myfile;
            myfile.open ("data.csv", std::ios_base::app);
            myfile << "\n" << blcksze << "," << sze << "," << blcks << "," << time ;
            myfile.close();
        }
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    int blcks = 0;
    int sze = 0;
    int blcksze = 0;
};





int doBenchmark(size_t const& N, int const& blocks, int const& blocksize, xpu::hd_buffer<float> &a, xpu::hd_buffer<float> &b, xpu::hd_buffer<float> &dst){
    
    xpu::initialize(xpu::driver::cuda);

    for (size_t i = 0; i < N; i++) {
        a.host()[i] = 2*i;
        b.host()[i] = 2*i+1;
    }

    xpu::copy(a, xpu::host_to_device);
    xpu::copy(b, xpu::host_to_device);
    Timer timer(N, blocks, blocksize);
    xpu::run_kernel<GpuMerge>(xpu::grid::n_blocks(blocks), a.device(), a.size(), b.device(), b.size(), dst.device());
    // for(int i = 10; i<=2048; i*=10)
    // {
    //     Timer timer(N, i,blocksize);
    //     xpu::run_kernel<GpuMerge>(xpu::grid::n_blocks(i), a.device(), a.size(), b.device(), b.size(), dst.device());
    // }
    // {
    //     Timer timer(N, 2000,blocksize);
    //     xpu::run_kernel<GpuMerge>(xpu::grid::n_blocks(2000), a.device(), a.size(), b.device(), b.size(), dst.device());
    // }

    xpu::copy(dst, xpu::device_to_host);

    float *h = dst.host();
    bool isSorted = true;
    for (size_t i = 1; i < dst.size(); i++) {
        isSorted &= (h[i-1] <= h[i]);
    }

    if(!isSorted){
        return -1;
    }

    // if (isSorted) {
    //     std::cout << "Data is sorted!" << std::endl;
    // } else {
    //     for (size_t i = 0; i < dst.size(); i++) {
    //         std::cout << h[i] << " ";
    //         if (i % 10 == 9) {
    //             std::cout << std::endl;
    //         }
    //     }
    //     std::cout << std::endl;
    //     std::cout << "ERROR: Data is not sorted!" << std::endl;
    // }

    return 0;

}



int main() {

    static constexpr size_t N0 = 1000;
    static constexpr int blocksize = 1024;
    static constexpr int blocks = 2000;

    xpu::initialize(xpu::driver::cuda);

    xpu::hd_buffer<float> a0{N0};
    xpu::hd_buffer<float> b0{N0};
    xpu::hd_buffer<float> dst0{a0.size() + b0.size()};

    doBenchmark(N0,blocks,blocksize,a0,b0,dst0);


    // static constexpr size_t N1 = 1000;

    // xpu::hd_buffer<float> a1{N1};
    // xpu::hd_buffer<float> b1{N1};
    // xpu::hd_buffer<float> dst1{a1.size() + b1.size()};

    // doBenchmark(N1,blocks,blocksize,a1,b1,dst1);


    // static constexpr size_t N2 = 10000;

    // xpu::initialize(xpu::driver::cuda);

    // xpu::hd_buffer<float> a2{N2};
    // xpu::hd_buffer<float> b2{N2};
    // xpu::hd_buffer<float> dst2{a2.size() + b2.size()};

    // doBenchmark(N2,blocks,blocksize,a2,b2,dst2);
    


    // static constexpr size_t N3 = 100000;

    // xpu::initialize(xpu::driver::cuda);

    // xpu::hd_buffer<float> a3{N3};
    // xpu::hd_buffer<float> b3{N3};
    // xpu::hd_buffer<float> dst3{a3.size() + b3.size()};

    // doBenchmark(N3,blocks,blocksize,a3,b3,dst3);
    


    // static constexpr size_t N4 = 1000000;

    // xpu::initialize(xpu::driver::cuda);

    // xpu::hd_buffer<float> a4{N4};
    // xpu::hd_buffer<float> b4{N4};
    // xpu::hd_buffer<float> dst4{a4.size() + b4.size()};

    // doBenchmark(N4,blocks,blocksize,a4,b4,dst4);
    
    return 0;
}
