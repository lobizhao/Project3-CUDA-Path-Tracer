#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

// Environment map sampling function
__device__ glm::vec3 Evaluate_EnvMap(Ray& r, cudaTextureObject_t envmapHandle) {
    if (envmapHandle == 0) return glm::vec3(0.0f);
    
    glm::vec3 dir = glm::normalize(r.direction);
    
    float phi = atan2f(dir.z, dir.x);
    float theta = asinf(dir.y);
    
    float u = (phi + 3.14159265f) / (2.0f * 3.14159265f);
    float v = 1.0f - (theta + 3.14159265f / 2.0f) / 3.14159265f;  
    
    float4 color = tex2D<float4>(envmapHandle, u, v);
    return glm::vec3(color.x, color.y, color.z);
}

//hw toggles
#define DEPTH_OF_FIELD 0
#define SORT_MATERIAL 0
#define COMPACTION 1
#define ANTI_ALIASING 1
#define RUSSIAN_ROULETTE 0
#define BETTER_RANDOM 1
#define USE_BVH 1


#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "bvh.h"

#define ERRORCHECK 0

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

struct IsAlive {
    __host__ __device__
        bool operator()(const PathSegment& s) const {
        return s.remainingBounces > 0;
    }
};

struct MaterialComparator {
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};

__host__ __device__ uint32_t fastHash(uint32_t seed) {
    seed ^= seed >> 16;
    seed *= 0x85ebca6b;
    seed ^= seed >> 13;
    seed *= 0xc2b2ae35;
    seed ^= seed >> 16;
    return seed;
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
#if BETTER_RANDOM
    uint32_t seed = index + (iter << 16) + (depth << 8);
    return thrust::default_random_engine(fastHash(seed));
#else
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
#endif
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Triangle* dev_triangles = NULL;
static LinearBVHNode* dev_bvhNodes = NULL;
static cudaTextureObject_t envmapHandle = 0;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    if (!scene->triangles.empty()) {
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    }
    
    // Load BVH to GPU
#if USE_BVH
    if (scene->bvhAccel && scene->bvhAccel->totalNodes > 0) {
        cudaMalloc(&dev_bvhNodes, scene->bvhAccel->totalNodes * sizeof(LinearBVHNode));
        cudaMemcpy(dev_bvhNodes, scene->bvhAccel->nodes, scene->bvhAccel->totalNodes * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);
    }
#endif
    
    // Load environment map to GPU
    envmapHandle = scene->envMap.loadToCuda();

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_triangles);
    cudaFree(dev_bvhNodes);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        
#if ANTI_ALIASING
        float jitterX = u01(rng);
        float jitterY = u01(rng);
#else
        float jitterX = 0.5f;
        float jitterY = 0.5f;
#endif
        
        // Calculate ray direction to focal plane
        glm::vec3 rayDir = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );
        
#if DEPTH_OF_FIELD
        // Calculate focal point
        glm::vec3 focalPoint = cam.position + cam.focalDistance * rayDir;
        
        // Sample point on lens (circular aperture)
        float theta = u01(rng) * 2.0f * 3.14159265f;
        float r = cam.lensRadius * sqrt(u01(rng));
        glm::vec3 lensOffset = r * (cos(theta) * cam.right + sin(theta) * cam.up);
        
        segment.ray.origin = cam.position + lensOffset;
        segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
#else
        segment.ray.origin = cam.position;
        segment.ray.direction = rayDir;
#endif
        
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// BVH-accelerated intersection function
__global__ void computeIntersectionsBVH(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    int triangles_size,
    LinearBVHNode* bvhNodes,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];
        Ray r = pathSegment.ray;

        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        bool outside = true;

        if (bvhNodes != nullptr) {
            // BVH traversal
            int stack[64];
            int ptr = 0;
            int bvhIdx = 0;

            while (true) {
                LinearBVHNode curNode = bvhNodes[bvhIdx];

                if (curNode.nPrimitives > 0) {
                    // Leaf node: test geometry
                    int geomID = curNode.geomID;
                    
                    glm::vec3 tmp_intersect, tmp_normal;
                    float t = -1.0f;
                    bool tmp_outside = true;
                    
                    if (geomID < geoms_size) {
                        // Test geometry
                        Geom& geom = geoms[geomID];
                        if (geom.type == CUBE) {
                            t = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_outside);
                        } else if (geom.type == SPHERE) {
                            t = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_outside);
                        }
                    } else {
                        // Test triangle
                        int triIdx = geomID - geoms_size;
                        if (triIdx >= 0 && triIdx < triangles_size) {
                            Triangle& triangle = triangles[triIdx];
                            t = triangleIntersectionTest(triangle, r, tmp_intersect, tmp_normal, tmp_outside);
                        }
                    }
                    
                    if (t > 0.0f && t < t_min) {
                        t_min = t;
                        hit_geom_index = geomID;
                        intersect_point = tmp_intersect;
                        normal = tmp_normal;
                        outside = tmp_outside;
                    }
                } else {
                    // Interior node: test children
                    int leftIndex = bvhIdx + 1;
                    int rightIndex = curNode.secondChildOffset;
                    
                    float leftHit = AABBIntersect(bvhNodes[leftIndex].bounds.pMin, bvhNodes[leftIndex].bounds.pMax, r);
                    float rightHit = AABBIntersect(bvhNodes[rightIndex].bounds.pMin, bvhNodes[rightIndex].bounds.pMax, r);
                    
                    // Traverse closer child first
                    if (leftHit >= 0.0f && rightHit >= 0.0f) {
                        if (leftHit > rightHit) {
                            bvhIdx = rightIndex;
                            stack[ptr++] = leftIndex;
                        } else {
                            bvhIdx = leftIndex;
                            stack[ptr++] = rightIndex;
                        }
                    } else if (leftHit >= 0.0f) {
                        bvhIdx = leftIndex;
                    } else if (rightHit >= 0.0f) {
                        bvhIdx = rightIndex;
                    } else {
                        if (ptr == 0) break;
                        bvhIdx = stack[--ptr];
                    }
                    continue;
                }
                
                // Pop from stack
                if (ptr == 0) break;
                bvhIdx = stack[--ptr];
            }
        }

        if (hit_geom_index == -1) {
            intersections[path_index].t = -1.0f;
        } else {
            intersections[path_index].t = t_min;
            if (hit_geom_index < geoms_size) {
                intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            } else {
                intersections[path_index].materialId = triangles[hit_geom_index - geoms_size].materialid;
            }
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// Original intersection function (fallback)
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    int triangles_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        // Test triangles
        for (int i = 0; i < triangles_size; i++)
        {
            Triangle& triangle = triangles[i];
            t = triangleIntersectionTest(triangle, pathSegment.ray, tmp_intersect, tmp_normal, outside);

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = geoms_size + i; // Offset by geoms count
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            if (hit_geom_index < geoms_size) {
                intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            } else {
                intersections[path_index].materialId = triangles[hit_geom_index - geoms_size].materialid;
            }
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

//replace new shadematerial

__global__ void shadeMaterial_with_BSDF(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    cudaTextureObject_t envmapHandle)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }
    PathSegment& path = pathSegments[idx];
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (path.remainingBounces <= 0) {
        return;
    }
    if (intersection.t > 0.0f)
    {
        Material material = materials[intersection.materialId];
        glm::vec3 intersect_point = path.ray.origin + intersection.t * path.ray.direction;

        if (material.emittance > 0.0f) {
            path.color *= (material.color * material.emittance);
            path.remainingBounces = 0;
        }
        else {
            path.remainingBounces--;
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, path.remainingBounces);
            scatterRay(path, intersect_point, intersection.surfaceNormal, material, rng);
        }
    }
    else {
        // Sample environment map for missed rays
        glm::vec3 envCol = Evaluate_EnvMap(path.ray, envmapHandle);
        path.color *= envCol;
        path.remainingBounces = 0;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing - optimized for memory coalescing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks


    //Part 1
    int num_paths = pixelcount;
    for (int depth = 0; depth < traceDepth; depth++)
    {
        if (num_paths == 0) break;

        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        
        // Use BVH if available, otherwise fall back to brute force
#if USE_BVH
        if (dev_bvhNodes != nullptr) {
            computeIntersectionsBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_triangles,
                hst_scene->triangles.size(),
                dev_bvhNodes,
                dev_intersections
                );
        } else {
#endif
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_triangles,
                hst_scene->triangles.size(),
                dev_intersections
                );
#if USE_BVH
        }
#endif
        checkCUDAError("trace one bounce");

        //replace shadeMaterial
#if SORT_MATERIAL
        if (depth == 0) {
            thrust::stable_sort_by_key(thrust::device,
                dev_intersections, dev_intersections + num_paths,
                dev_paths, MaterialComparator());
        }
#endif

        shadeMaterial_with_BSDF << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_geoms,
            envmapHandle
            );
        checkCUDAError("shade and scatter");

#if COMPACTION
        if (depth % 2 == 1 || depth == traceDepth - 1) {
            auto lastPath = dev_paths + num_paths;
            auto mid = thrust::stable_partition(thrust::device, dev_paths, lastPath, IsAlive{});
            num_paths = mid - dev_paths;
        }
#endif 

        if (guiData != NULL) {
            guiData->TracedDepth = depth + 1;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Only copy to CPU occasionally for debugging (major performance killer)
    // cudaMemcpy(hst_scene->state.image.data(), dev_image,
    //     pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}