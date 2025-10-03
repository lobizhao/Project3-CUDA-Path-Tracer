#pragma once

#include "sceneStructs.h"
#include <glm/glm.hpp>
#include <vector>
#include <memory>

struct Bounds3f {
    glm::vec3 pMin, pMax;
    
    __host__ __device__ Bounds3f() {
        float minNum = std::numeric_limits<float>::lowest();
        float maxNum = std::numeric_limits<float>::max();
        pMin = glm::vec3(maxNum);
        pMax = glm::vec3(minNum);
    }
    
    __host__ __device__ Bounds3f(const glm::vec3& p) : pMin(p), pMax(p) {}
    
    __host__ __device__ Bounds3f(const glm::vec3& p1, const glm::vec3& p2) 
        : pMin(glm::min(p1, p2)), pMax(glm::max(p1, p2)) {}
    
    __host__ __device__ glm::vec3 Diagonal() const { return pMax - pMin; }
    
    __host__ __device__ float SurfaceArea() const {
        glm::vec3 d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
    
    __host__ __device__ int MaximumExtent() const {
        glm::vec3 d = Diagonal();
        if (d.x > d.y && d.x > d.z) return 0;
        else if (d.y > d.z) return 1;
        else return 2;
    }
    
    __host__ __device__ glm::vec3 Centroid() const { return 0.5f * (pMin + pMax); }
};

__host__ __device__ inline Bounds3f Union(const Bounds3f& b, const glm::vec3& p) {
    Bounds3f ret;
    ret.pMin = glm::min(b.pMin, p);
    ret.pMax = glm::max(b.pMax, p);
    return ret;
}

__host__ __device__ inline Bounds3f Union(const Bounds3f& b1, const Bounds3f& b2) {
    Bounds3f ret;
    ret.pMin = glm::min(b1.pMin, b2.pMin);
    ret.pMax = glm::max(b1.pMax, b2.pMax);
    return ret;
}

struct Primitive {
    int geomID;
    Bounds3f bounds;
    glm::vec3 centroid;
    
    Primitive(int id, const Bounds3f& b) : geomID(id), bounds(b), centroid(b.Centroid()) {}
};

struct LinearBVHNode {
    Bounds3f bounds;
    union {
        int geomID;              // Leaf: geometry ID
        int secondChildOffset;   // Interior: right child offset
    };
    int nPrimitives;            // >0 for leaf nodes
    int axis;                   // Split axis for interior nodes
};

struct BVHBuildNode {
    Bounds3f bounds;
    BVHBuildNode* children[2];
    int splitAxis, firstPrimOffset, nPrimitives;
    
    BVHBuildNode() {
        children[0] = children[1] = nullptr;
        nPrimitives = 0;
    }
    
    void InitLeaf(int first, int n, const Bounds3f& b) {
        firstPrimOffset = first;
        nPrimitives = n;
        bounds = b;
        children[0] = children[1] = nullptr;
    }
    
    void InitInterior(int axis, BVHBuildNode* c0, BVHBuildNode* c1) {
        children[0] = c0;
        children[1] = c1;
        bounds = Union(c0->bounds, c1->bounds);
        splitAxis = axis;
        nPrimitives = 0;
    }
};

class BVHAccel {
public:
    BVHAccel(std::vector<std::shared_ptr<Primitive>>& primitives, int maxPrimsInNode = 1);
    ~BVHAccel();
    
    LinearBVHNode* nodes;
    int totalNodes;
    std::vector<std::shared_ptr<Primitive>> primitives;

private:
    BVHBuildNode* recursiveBuild(std::vector<Primitive>& primitiveInfo, 
                                int start, int end, int* totalNodes,
                                std::vector<std::shared_ptr<Primitive>>& orderedPrims);
    int flattenBVHTree(BVHBuildNode* node, int* offset);
    
    int maxPrimsInNode;
};

// GPU intersection function (implemented in intersections.cu)
__host__ __device__ float AABBIntersect(const glm::vec3& pMin, const glm::vec3& pMax, const Ray& r);