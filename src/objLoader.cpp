#include "objLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<Triangle> OBJLoader::loadOBJ(const std::string& filename, int materialId) {
    std::vector<Triangle> triangles;
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open OBJ file: " << filename << std::endl;
        return triangles;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        
        if (prefix == "v") {
            glm::vec3 vertex;
            iss >> vertex.x >> vertex.y >> vertex.z;
            vertices.push_back(vertex);
        }
        else if (prefix == "vn") {
            glm::vec3 normal;
            iss >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
        }
        else if (prefix == "f") {
            std::string v1, v2, v3;
            iss >> v1 >> v2 >> v3;
        
            auto parseIndex = [](const std::string& str) -> std::pair<int, int> {
                size_t pos1 = str.find('/');
                size_t pos2 = str.find('/', pos1 + 1);
                int vIdx = std::stoi(str.substr(0, pos1)) - 1;
                int nIdx = std::stoi(str.substr(pos2 + 1)) - 1;
                return std::make_pair(vIdx, nIdx);
            };
            
            std::pair<int, int> p1 = parseIndex(v1);
            std::pair<int, int> p2 = parseIndex(v2);
            std::pair<int, int> p3 = parseIndex(v3);
            
            int v1Idx = p1.first, n1Idx = p1.second;
            int v2Idx = p2.first, n2Idx = p2.second;
            int v3Idx = p3.first, n3Idx = p3.second;
            
            Triangle tri;
            tri.v0 = vertices[v1Idx];
            tri.v1 = vertices[v2Idx];
            tri.v2 = vertices[v3Idx];
            tri.n0 = normals[n1Idx];
            tri.n1 = normals[n2Idx];
            tri.n2 = normals[n3Idx];
            tri.materialid = materialId;
            
            triangles.push_back(tri);
        }
    }
    
    //std::cout << "Loaded " << triangles.size() << " triangles from " << filename << std::endl;
    return triangles;
}