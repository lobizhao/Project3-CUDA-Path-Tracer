#include "bvh.h"
#include <algorithm>
#include <iostream>

struct BucketInfo {
    int count = 0;
    Bounds3f bounds;
};

BVHAccel::BVHAccel(std::vector<std::shared_ptr<Primitive>>& prims, int maxPrimsInNode)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), primitives(std::move(prims)) {
    
    if (primitives.empty()) {
        nodes = nullptr;
        return;
    }
    
    // Build BVH from primitives
    std::vector<Primitive> primitiveInfo;
    primitiveInfo.reserve(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i) {
        primitiveInfo.emplace_back(i, primitives[i]->bounds);
    }
    
    std::vector<std::shared_ptr<Primitive>> orderedPrims;
    orderedPrims.reserve(primitives.size());
    totalNodes = 0;
    
    BVHBuildNode* root = recursiveBuild(primitiveInfo, 0, primitives.size(), 
                                       &totalNodes, orderedPrims);
    primitives.swap(orderedPrims);
    
    // Allocate and flatten BVH
    nodes = new LinearBVHNode[totalNodes];
    int offset = 0;
    flattenBVHTree(root, &offset);
}

BVHAccel::~BVHAccel() {
    delete[] nodes;
}

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Primitive>& primitiveInfo,
                                      int start, int end, int* totalNodes,
                                      std::vector<std::shared_ptr<Primitive>>& orderedPrims) {
    BVHBuildNode* node = new BVHBuildNode;
    (*totalNodes)++;
    
    // Compute bounds of all primitives
    Bounds3f bounds;
    for (int i = start; i < end; ++i) {
        bounds = Union(bounds, primitiveInfo[i].bounds);
    }
    
    int nPrimitives = end - start;
    if (nPrimitives == 1) {
        // Create leaf node
        int firstPrimOffset = orderedPrims.size();
        for (int i = start; i < end; ++i) {
            int primNum = primitiveInfo[i].geomID;
            orderedPrims.push_back(primitives[primNum]);
        }
        node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
        return node;
    }
    
    // Compute bound of primitive centroids
    Bounds3f centroidBounds;
    for (int i = start; i < end; ++i) {
        centroidBounds = Union(centroidBounds, primitiveInfo[i].centroid);
    }
    int dim = centroidBounds.MaximumExtent();
    
    // Partition primitives into two sets and build children
    int mid = (start + end) / 2;
    if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
        // Create leaf node if centroids are at same position
        int firstPrimOffset = orderedPrims.size();
        for (int i = start; i < end; ++i) {
            int primNum = primitiveInfo[i].geomID;
            orderedPrims.push_back(primitives[primNum]);
        }
        node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
        return node;
    }
    
    // SAH partition
    if (nPrimitives <= 2) {
        // Simple partition for small number of primitives
        std::nth_element(&primitiveInfo[start], &primitiveInfo[mid], &primitiveInfo[end-1]+1,
                        [dim](const Primitive& a, const Primitive& b) {
                            return a.centroid[dim] < b.centroid[dim];
                        });
    } else {
        // SAH with buckets
        constexpr int nBuckets = 12;
        BucketInfo buckets[nBuckets];
        
        // Initialize buckets
        for (int i = start; i < end; ++i) {
            int b = nBuckets * ((primitiveInfo[i].centroid[dim] - centroidBounds.pMin[dim]) /
                               (centroidBounds.pMax[dim] - centroidBounds.pMin[dim]));
            if (b == nBuckets) b = nBuckets - 1;
            buckets[b].count++;
            buckets[b].bounds = Union(buckets[b].bounds, primitiveInfo[i].bounds);
        }
        
        // Compute costs for splitting after each bucket
        float cost[nBuckets - 1];
        for (int i = 0; i < nBuckets - 1; ++i) {
            Bounds3f b0, b1;
            int count0 = 0, count1 = 0;
            for (int j = 0; j <= i; ++j) {
                b0 = Union(b0, buckets[j].bounds);
                count0 += buckets[j].count;
            }
            for (int j = i + 1; j < nBuckets; ++j) {
                b1 = Union(b1, buckets[j].bounds);
                count1 += buckets[j].count;
            }
            cost[i] = 1 + (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea()) / bounds.SurfaceArea();
        }
        
        // Find bucket to split at that minimizes SAH metric
        float minCost = cost[0];
        int minCostSplitBucket = 0;
        for (int i = 1; i < nBuckets - 1; ++i) {
            if (cost[i] < minCost) {
                minCost = cost[i];
                minCostSplitBucket = i;
            }
        }
        
        // Split primitives at selected SAH bucket
        Primitive* pmid = std::partition(&primitiveInfo[start], &primitiveInfo[end-1]+1,
                                       [=](const Primitive& pi) {
                                           int b = nBuckets * ((pi.centroid[dim] - centroidBounds.pMin[dim]) /
                                                              (centroidBounds.pMax[dim] - centroidBounds.pMin[dim]));
                                           if (b == nBuckets) b = nBuckets - 1;
                                           return b <= minCostSplitBucket;
                                       });
        mid = pmid - &primitiveInfo[0];
    }
    
    node->InitInterior(dim,
                      recursiveBuild(primitiveInfo, start, mid, totalNodes, orderedPrims),
                      recursiveBuild(primitiveInfo, mid, end, totalNodes, orderedPrims));
    return node;
}

int BVHAccel::flattenBVHTree(BVHBuildNode* node, int* offset) {
    LinearBVHNode* linearNode = &nodes[*offset];
    linearNode->bounds = node->bounds;
    int myOffset = (*offset)++;
    
    if (node->nPrimitives > 0) {
        // Leaf node - store the actual geometry ID from the first primitive
        linearNode->geomID = primitives[node->firstPrimOffset]->geomID;
        linearNode->nPrimitives = node->nPrimitives;
    } else {
        // Interior node
        linearNode->axis = node->splitAxis;
        linearNode->nPrimitives = 0;
        flattenBVHTree(node->children[0], offset);
        linearNode->secondChildOffset = flattenBVHTree(node->children[1], offset);
    }
    return myOffset;
}

