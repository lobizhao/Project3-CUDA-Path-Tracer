#pragma once

#include "sceneStructs.h"
#include <vector>
#include <string>

class OBJLoader {
public:
    static std::vector<Triangle> loadOBJ(const std::string& filename, int materialId);
};