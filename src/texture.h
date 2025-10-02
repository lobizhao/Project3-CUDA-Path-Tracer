#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>

class Texture {
public:
    int w = 0, h = 0, c = 0;
    bool isHDR = false;
    
    std::vector<unsigned char> cpudata;
    std::vector<float> cpudataHDR;
    
    cudaArray_t array = nullptr;
    cudaTextureObject_t handle = 0;
    
    void loadToCPU(const std::string& filename);
    void loadToCPU(unsigned char* data, int w, int h, int c);
    cudaTextureObject_t loadToCuda();
    void FreeCudaSide();
    
    ~Texture();
};