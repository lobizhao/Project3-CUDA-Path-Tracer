#include "scene.h"
#include "objLoader.h"
#include "bvh.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>
#include <chrono>

using namespace std;
using json = nlohmann::json;

// Helper function to compute bounds for geometry
Bounds3f computeGeomBounds(const Geom& geom) {
    Bounds3f bounds;
    if (geom.type == SPHERE) {
        // For sphere, compute bounds in world space
        glm::vec3 center = glm::vec3(geom.transform * glm::vec4(0, 0, 0, 1));
        float radius = glm::length(glm::vec3(geom.transform * glm::vec4(0.5f, 0, 0, 0)));
        bounds.pMin = center - glm::vec3(radius);
        bounds.pMax = center + glm::vec3(radius);
    } else if (geom.type == CUBE) {
        // For cube, transform corners and compute bounds
        glm::vec3 corners[8] = {
            {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f},
            {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f},
            {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f},
            {-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}
        };
        bounds = Bounds3f();
        for (int i = 0; i < 8; ++i) {
            glm::vec3 worldPos = glm::vec3(geom.transform * glm::vec4(corners[i], 1.0f));
            bounds = Union(bounds, worldPos);
        }
    }
    return bounds;
}

Bounds3f computeTriangleBounds(const Triangle& tri) {
    Bounds3f bounds(tri.v0);
    bounds = Union(bounds, tri.v1);
    bounds = Union(bounds, tri.v2);
    return bounds;
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 0.0f;
            newMaterial.hasRefractive = 0.0f;
            newMaterial.emittance = 0.0f;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.hasReflective = 0.0f;
            newMaterial.hasRefractive = 0.0f;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
            newMaterial.hasRefractive = 0.0f;
            newMaterial.emittance = 0.0f;
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.hasReflective = 0.0f;
            newMaterial.emittance = 0.0f;
            newMaterial.indexOfRefraction = p["REFRACTION"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        
        if (type == "mesh")
        {
            // Load OBJ mesh
            std::string filename = p["FILE"];
            int materialId = MatNameToID[p["MATERIAL"]];
            auto meshTriangles = OBJLoader::loadOBJ(filename, materialId);
            
            // Apply transformations to triangles
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
            glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            glm::vec3 scaleVec = glm::vec3(scale[0], scale[1], scale[2]);
            
            glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, scaleVec);
            glm::mat3 normalMat = glm::mat3(glm::transpose(glm::inverse(transform)));
            
            for (auto& tri : meshTriangles) {
                tri.v0 = glm::vec3(transform * glm::vec4(tri.v0, 1.0f));
                tri.v1 = glm::vec3(transform * glm::vec4(tri.v1, 1.0f));
                tri.v2 = glm::vec3(transform * glm::vec4(tri.v2, 1.0f));
                tri.n0 = glm::normalize(normalMat * tri.n0);
                tri.n1 = glm::normalize(normalMat * tri.n1);
                tri.n2 = glm::normalize(normalMat * tri.n2);
                triangles.push_back(tri);
            }
            continue;
        }
        
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //for part 2 - 02 depth of field
    camera.lensRadius = cameraData.contains("LENS_RADIUS") ? (float)cameraData["LENS_RADIUS"] : 0.1f;
    camera.focalDistance = cameraData.contains("FOCAL_DISTANCE") ? (float)cameraData["FOCAL_DISTANCE"] : glm::length(camera.lookAt - camera.position);
    
    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    
    // Load environment map if specified
    if (data.contains("EnvMap")) {
        const auto& envMapData = data["EnvMap"];
        const auto& envmap_path = envMapData["PATH"];
        std::string fullenvpath = envmap_path.get<std::string>();
        envMap.loadToCPU(fullenvpath);
        std::cout << "Loaded environment map: " << fullenvpath << std::endl;
    }
    
    // Build BVH
    std::vector<std::shared_ptr<Primitive>> primitives;
    for (int i = 0; i < geoms.size(); ++i) {
        Bounds3f bounds = computeGeomBounds(geoms[i]);
        primitives.push_back(std::make_shared<Primitive>(i, bounds));
    }
    for (int i = 0; i < triangles.size(); ++i) {
        Bounds3f bounds = computeTriangleBounds(triangles[i]);
        primitives.push_back(std::make_shared<Primitive>(geoms.size() + i, bounds));
    }
    if (!primitives.empty()) {
        auto start = std::chrono::high_resolution_clock::now();
        bvhAccel = std::make_unique<BVHAccel>(primitives, 1);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Built BVH with " << bvhAccel->totalNodes << " nodes for " 
                  << primitives.size() << " primitives in " << duration.count() << "ms" << std::endl;
    }
}
