#define STB_IMAGE_WRITE_IMPLEMENTATION
#define EPSILON 0.001
#include "stb_image_write.h"
#include <cmath>
#include <array>
#include <vector>
#include <cfloat>
#include <cassert>
#include <iostream>

// simple cpu raytracer - jaysen jaehnig
// references: 
// https://github.com/nothings/stb, 
// https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
// https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
// https://www.sunshine2k.de/articles/coding/vectorreflection/vectorreflection.html

enum Projection {
    ORTHOGRAPHIC,
    PERSPECTIVE
};

struct Vec3 {
    float x, y, z;

    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3() : x(0), y(0), z(0) {}

    // Addition
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    // Subtraction
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    // Scalar multiplication
    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    // Scalar division
    Vec3 operator/(float scalar) const {
        if (scalar == 0) throw std::invalid_argument("Division by zero");
        return Vec3(x / scalar, y / scalar, z / scalar);
    }

    // Dot product
    float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // Cross product
    Vec3 cross(const Vec3& other) const {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    // Magnitude
    float magnitude() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    // Normalization
    Vec3 normalize() const {
        float mag = magnitude();
        if (mag == 0) throw std::invalid_argument("Cannot normalize a zero vector");
        return *this / mag;
    }

    // Print (for debugging)
    void print() const {
        std::cout << "Vec3(" << x << ", " << y << ", " << z << ")" << std::endl;
    }
};

class Ray {
public:
    Ray(const Vec3 &origin, const Vec3 &destination)
        : origin(origin)
    {
        direction = ( destination - origin ).normalize();
    }

    Vec3 getOrigin() const { return origin; }
    Vec3 getDirection() const { return direction; }
    void setOrigin(const Vec3 &newOrigin) { origin = newOrigin; }
    void setDirection(const Vec3 &newDirection) { direction = newDirection; }

private:
    Vec3 origin;
    Vec3 direction;
};

class Material { // 0-255 color representation
public:
    Material(unsigned char r, unsigned char g, unsigned char b, unsigned char a, bool reflective)
        : r(r), g(g), b(b), a(a), reflective(reflective) {}
    Material(): r(0), g(0), b(0), a(1) {}

    unsigned char getR() const { return r; }
    unsigned char getG() const { return g; }
    unsigned char getB() const { return b; }
    unsigned char getA() const { return a; }
    bool isReflective() const { return reflective; }

    void setR(unsigned char newR) { r = newR; }
    void setG(unsigned char newG) { g = newG; }
    void setB(unsigned char newB) { b = newB; }
    void setA(unsigned char newA) { a = newA; }
    void setReflective(bool newReflective) { reflective = newReflective; }

private:
    unsigned char r, g, b, a;
    bool reflective;
};

class Rayhit {
public:
    Rayhit(float t, const Material &material, const Vec3 &pos, const Vec3 &normal) 
        : t(t), material(material), pos(pos), normal(normal) {}

    // Constructor with only t (default-initialize other members) used for when ray hits nothing
    Rayhit(float t) 
        : t(t), material(Material()), pos(Vec3()), normal(Vec3()) {}

    float getT() const { return t; }
    const Material& getMaterial() const { return material; }
    const Vec3& getPos() const { return pos; }
    const Vec3& getNormal() const { return normal; }

    void setT(float newT) { t = newT; }
    void setMaterial(const Material &newMaterial) { material = newMaterial; }
    void setPos(const Vec3 &newPos) { pos = newPos; }
    void setNormal(const Vec3 &newNormal) { normal = newNormal; }

private:
    float t;
    Material material;
    Vec3 pos; // intersect position
    Vec3 normal; // normal vector of hit surface
};

class Sphere {
public:
    Sphere(const Vec3& inputCenter, float radius, const Material& material)
        : center(inputCenter), radius(radius), material(material) {}
    ~Sphere() {}

    const Vec3& getCenter() const { return center; }
    float getRadius() const { return radius; }
    const Material& getMaterial() const { return material; }

    void setCenter(const Vec3 &newCenter) { center = newCenter; }
    void setRadius(float newRadius) { radius = newRadius; }
    void setMaterial(const Material &newMaterial) { material = newMaterial; }

    float intersect(const Ray& ray) const {
        // referenced math from https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        const Vec3& o = ray.getOrigin();
        const Vec3& d = ray.getDirection();
        Vec3 oc = o - center;

        float a = d.dot(d);
        float b = 2.0f * oc.dot(d);
        float c = oc.dot(oc) - radius * radius;

        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            return -1.0f; // No intersection
        }

        // Compute the two intersection points
        float t1 = (-b - std::sqrt(discriminant)) / (2.0f * a);
        float t2 = (-b + std::sqrt(discriminant)) / (2.0f * a);

        return (t1 > 0) ? t1 : (t2 > 0) ? t2 : -1.0f;
    }

private:
    Vec3 center;
    float radius;
    Material material;
};

class Triangle {
public:
    Triangle(const std::array<Vec3, 3>& inputVertices, const Material& material)
        : vertices(inputVertices), material(material) {
            Vec3 vec1 = (vertices[1] - vertices[0]);
            Vec3 vec2 = (vertices[2] - vertices[0]);

            normal = vec1.cross(vec2).normalize();
        }

    const std::array<Vec3, 3>& getVertices() const { return vertices; }
    const Material& getMaterial() const { return material; }
    const Vec3& getNormal() const { return normal; }

    void setVertices(const std::array<Vec3, 3> &newVertices) { vertices = newVertices; }
    void setMaterial(const Material &newMaterial) { material = newMaterial; }
    void setNormal(const Vec3 &newNormal) { normal = newNormal; }

    float intersect(const Ray &ray) const {
        // referenced math from https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf

        Vec3 vec1 = vertices[1] - vertices[0];
        Vec3 vec2 = vertices[2] - vertices[0];
        Vec3 tvec = ray.getOrigin() - vertices[0];
        Vec3 pvec = ray.getDirection().cross(vec2);
        Vec3 qvec = tvec.cross(vec1);

        // Determinant
        float det = vec1.dot(pvec);
        if (det > -EPSILON && det < EPSILON) 
            return -1;
        float invDet = 1.0f / det;

        // Calculate u parameter
        float u = tvec.dot(pvec) * invDet;
        if (u < 0 || u > 1) 
            return -1;

        // Calculate v parameter
        float v = ray.getDirection().dot(qvec) * invDet;
        if (v < 0 || u + v > 1) 
            return -1;

        // Calculate t
        float t = vec2.dot(qvec) * invDet;
        return t < 0 ? -1 : t;
    }

private:
    std::array<Vec3, 3> vertices;
    Vec3 normal;
    Material material;
};

class Raytracer {
public:
    Raytracer( const float width, const float height, const Vec3 &camPos, const Vec3 &lightPos) 
        : width(width), height(height), camPos(camPos), lightPos(lightPos)
        {
            frameBuffer.resize(width*height*3);
            
            Material blue = Material(0,0,255, 255, false);
            Material white = Material(255,255,255, 255, false);
            Material red = Material(255,0,0,255, false);
            Material refl = Material(255,255,255,255, true);

            // back wall
            triangles.emplace_back(Triangle(std::array<Vec3, 3>{ Vec3{-8,-2,-20}, Vec3{8,-2,-20}, Vec3{8,10,-20} }, blue ));
            triangles.emplace_back(Triangle(std::array<Vec3, 3>{ Vec3{-8,-2,-20}, Vec3{8,10,-20}, Vec3{-8,10,-20} }, blue ));
            // floor
            triangles.emplace_back(Triangle(std::array<Vec3, 3>{ Vec3{-8,-2,-20}, Vec3{8,-2,-10}, Vec3{8,-2,-20} }, white ));
            triangles.emplace_back(Triangle(std::array<Vec3, 3>{ Vec3{-8,-2,-20}, Vec3{-8,-2,-10}, Vec3{8,-2,-10} }, white ));
            // right red triangle
            triangles.emplace_back(Triangle(std::array<Vec3, 3>{ Vec3{8,-2,-20}, Vec3{8,-2,-10}, Vec3{8,10,-20} }, red ));
            
            spheres.emplace_back(Sphere(Vec3{0,0,-16}, 2, refl));
            spheres.emplace_back(Sphere(Vec3{3,-1,-14}, 1, refl));
            spheres.emplace_back(Sphere(Vec3{-3,-1,-14}, 1, red));

        }
    
    void render(Projection projection) {
        // cast ray for every pixel
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {

                // calculate frame buffer index for current pixel
                int frameBufIndex = j * num_channels + i * width * num_channels;
                Vec3 pixelPos = pixelPos2Dto3D(j,i);

                // used for orthographic projection
                Vec3 rayDest = {pixelPos.x, pixelPos.y, pixelPos.z - 1};
                enum Projection ortho = ORTHOGRAPHIC;

                Ray ray = (projection == ORTHOGRAPHIC) ? Ray(pixelPos, rayDest) : Ray(camPos, pixelPos);
                Rayhit hit = geometryIntersect(ray);
                if (hit.getT() > 0) {
                    assert(hit.getMaterial().getR() <= 255 && hit.getMaterial().getR() >= 0 );
                }

                // handle reflections math referenced from https://www.sunshine2k.de/articles/coding/vectorreflection/vectorreflection.html
                int k = 0;
                int depth = 10;
                while (k < depth && hit.getT() > 0 && hit.getMaterial().isReflective()) 
                { 
                    float dot = ray.getDirection().dot(hit.getNormal());
                    Vec3 vec = ray.getDirection() + hit.getNormal() * (-2 * dot) ;
                    ray = Ray( hit.getPos() + vec * EPSILON, hit.getPos() + vec );
                    hit = geometryIntersect(ray);
                    k++;
                }
               
                if ( hit.getT() > 0) { 
                    // calculate diffuse shading (clamp negative values to 0)
                    Ray lightRay = Ray(hit.getPos(), lightPos);
                    float diffuse = hit.getNormal().dot(lightRay.getDirection());
                    if (diffuse < 0.2f)
                        diffuse = 0.2f;
                    
                    // calculate shadow
                    lightRay.setOrigin(hit.getPos() + (lightRay.getDirection() * EPSILON));
                    Rayhit shadowHit = geometryIntersect(lightRay);
                    if (shadowHit.getT() > 0) {
                        diffuse = 0.2f;
                    }
                    
                    // set color of pixel in frame buffer
                    frameBuffer[frameBufIndex] = hit.getMaterial().getR()*diffuse;
                    frameBuffer[frameBufIndex+1] = hit.getMaterial().getG()*diffuse;
                    frameBuffer[frameBufIndex+2] = hit.getMaterial().getB()*diffuse;
                }
                else {
                    // set color of pixel in frame buffer to black (no hit)
                    frameBuffer[frameBufIndex] = 0;
                    frameBuffer[frameBufIndex+1] = 0;
                    frameBuffer[frameBufIndex+2] = 0;
                }
            }
        }
        // write frame buffer to a png
        stbi_write_png("reference.png", width, height, num_channels, frameBuffer.data(), width * num_channels);
    }

private:
    Vec3 camPos;
    Vec3 lightPos;
    float width;
    float height;
    float num_channels = 3;
    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
    std::vector<unsigned char> frameBuffer;

    Rayhit geometryIntersect(const Ray &ray) 
    {
        float minT = FLT_MAX;
        Rayhit hit = Rayhit(-1);

        // check spheres 
        for (const auto& sphere : spheres) 
        {
            float t = sphere.intersect(ray);
            if (t < minT && t > 0) 
            {
                minT = t;
                Vec3 pos = ray.getOrigin() + ray.getDirection() * t;
                Vec3 normal = (pos - sphere.getCenter()).normalize();
                hit = Rayhit(t, sphere.getMaterial(), pos, normal);
            }
        }
        // check triangles 
        for (const auto& triangle : triangles) 
        {
            float t = triangle.intersect(ray);
            if (t < minT && t > 0) {
                minT = t;
                Vec3 pos = pos = ray.getOrigin() + ray.getDirection() * t;
                hit = Rayhit(t, triangle.getMaterial(), pos, triangle.getNormal());
            }     
        }
        return hit;
    }

    Vec3 pixelPos2Dto3D(float x, float y)
    {
        float aspectRatio = width/height;
        float pixelWidth = (2.0f * aspectRatio)/width;
        float pixelHeight = (2.0f/height);
        // Convert x,y (pixel coordinate) to x,y,z coordinate on image plane;
        return { 
            ( -aspectRatio +  pixelWidth/2.0f + pixelWidth*x ), 
            (1.0f - ( pixelHeight/2.0f + pixelHeight*y) ),
            -2.0f 
        };
    }

};

int main(int argc, char** argv) {
    Raytracer raytracer = Raytracer(1920, 1080, {0, 0, 0}, {3, 5, -15});
    Projection p = PERSPECTIVE;
    raytracer.render(p);
    return 0;
}


