#pragma once

#include "sceneStructs.h"
#include "texture.h"
#include "bvh.h"
#include <vector>
#include <memory>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    RenderState state;
    Texture envMap;
    std::unique_ptr<BVHAccel> bvhAccel;
};
