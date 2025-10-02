#include "sceneStructs.h"
#include <cuda_runtime.h>

__device__ glm::vec3 Evaluate_EnvMap(Ray& r, cudaTextureObject_t envmapHandle) {
    if (envmapHandle == 0) return glm::vec3(0.0f);
    
    // Convert ray direction to spherical coordinates
    glm::vec3 dir = glm::normalize(r.direction);
    
    // Calculate UV coordinates for equirectangular mapping
    float phi = atan2f(dir.z, dir.x);
    float theta = asinf(dir.y);
    
    float u = (phi + M_PI) / (2.0f * M_PI);
    float v = (theta + M_PI / 2.0f) / M_PI;
    
    // Sample HDR texture
    float4 color = tex2D<float4>(envmapHandle, u, v);
    return glm::vec3(color.x, color.y, color.z);
}