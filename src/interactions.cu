#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ float schlickFresnel(float cosTheta, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow(1.0f - cosTheta, 5.0f);
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    if (m.hasRefractive > 0.0f) {
        // Refractive material (glass)
        glm::vec3 incident = pathSegment.ray.direction;
        float cosTheta = glm::dot(-incident, normal);
        bool entering = cosTheta > 0;
        
        float eta = entering ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;
        glm::vec3 n = entering ? normal : -normal;
        cosTheta = abs(cosTheta);
        
        // Fresnel reflection probability (use material IOR, not eta)
        float fresnel = schlickFresnel(cosTheta, m.indexOfRefraction);
        
        if (u01(rng) < fresnel) {
            // Reflection
            pathSegment.ray.direction = glm::reflect(incident, n);
            pathSegment.ray.origin = intersect + 0.001f * n;
        } else {
            // Refraction
            glm::vec3 refracted = glm::refract(incident, n, eta);
            if (glm::length(refracted) < 0.001f) {
                // Total internal reflection
                pathSegment.ray.direction = glm::reflect(incident, n);
                pathSegment.ray.origin = intersect + 0.001f * n;
            } else {
                pathSegment.ray.direction = refracted;
                pathSegment.ray.origin = intersect - 0.001f * n;
            }
        }
    } else if (m.hasReflective > 0.0f) {
        // Perfect specular reflection
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect + 0.001f * normal;
    } else {
        // Diffuse material
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect + 0.001f * normal;
    }
    
    pathSegment.color *= m.color;
}