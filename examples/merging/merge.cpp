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


class timeToFile{
public:
    timeToFile() = default;
    ~timeToFile() = default;
    
    void writeTime(int const& size, int const& blocks, int const& blocksize, float &time){
        printf("blck%i        %fms        %i        %i\n", blocksize, time, size, blocks);
        std::ofstream myfile;
        myfile.open ("data.csv", std::ios_base::app);
        myfile << "\nParallelMerge" << "," << size << "," << blocksize << "," << time ;
        myfile.close();
    }
};





int doBenchmark(size_t const& N, int const& blocks, xpu::hd_buffer<float> &a, xpu::hd_buffer<float> &b, xpu::hd_buffer<float> &dst){
    
    xpu::initialize(xpu::driver::cuda);
    int prod = log10(N);
    for (size_t i = 0; i < N; i++) {
        a.host()[i] = prod*i;
        b.host()[i] = prod*i+1;
    }

    xpu::copy(a, xpu::host_to_device);
    xpu::copy(b, xpu::host_to_device);
    std::vector<float> time;
    {
        //Timer timer(N, blocks, blocksize);
        xpu::run_kernel<GpuMerge>(xpu::grid::n_blocks(blocks), a.device(), a.size(), b.device(), b.size(), dst.device());
        
    }
    

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
        //std::cout << h[i] << std::endl;
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

    timeToFile ttf;

    xpu::initialize(xpu::driver::cuda);

    static constexpr size_t N0 = 10;
    static constexpr size_t N1 = 100;
    static constexpr size_t N2 = 1000;
    static constexpr size_t N3 = 10000;
    static constexpr size_t N4 = 100000;
    static constexpr size_t N5 = 1000000;

    static constexpr int blocksize = 64;
    static constexpr int blocks = 2000;

    auto dodo = [&](size_t const& N, bool warmUp){
        xpu::hd_buffer<float> a0{N};
        xpu::hd_buffer<float> b0{N};
        xpu::hd_buffer<float> dst0{a0.size() + b0.size()};

        auto ret = doBenchmark(N,blocks,a0,b0,dst0);
        if(ret != 0){
            std:: cout << "not sorted" << std::endl;
        }
        if(!warmUp){
            auto time = xpu::get_timing<GpuMerge>();
            ttf.writeTime(N,blocks, blocksize, time[time.size()-1]);
        }
        return;
    };

    //WarmUp
    dodo(N0,true);

    for(int i = 0; i<20; i++){
    //BenchMarks
        dodo(N0,false);
        dodo(N1,false);    
        dodo(N2, false);
        dodo(N3, false);
        dodo(N4, false);
        dodo(N5, false);
    }
        
    return 0;
}
