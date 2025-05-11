
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <unistd.h>
#include <random>
#include <algorithm>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Eigen::Vector3d;
using Eigen::Vector2d; 
using Eigen::Matrix4d;
using Eigen::Vector4d;

/*OVERLOADERS*/

std::ostream& operator<<(std::ostream& os, const Vector3d& vec) {
    os << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
    return os;
}

//*************all classes/structures below**********************
//to keep track of multiple material color values
template<typename MatrixType>
void printMatrix(const MatrixType& matrix, const std::string& name) {
    std::cout << name << ":\n" << matrix << "\n";
}

class DepthBuffer {
public:
    std::vector<double> buffer;
    int width, height;

    DepthBuffer(int w, int h, double initVal = INFINITY)
        : width(w), height(h), buffer(w * h, initVal) {}

    double get(int x, int y) const {
    return buffer[y * width + x];
}

void set(int x, int y, double value) {
    buffer[y * width + x] = value;
}

    // Clear the depth buffer by resetting all values to initVal
    void clear(double initVal = INFINITY) {
        std::fill(buffer.begin(), buffer.end(), initVal);
    }
};

struct Material {
    Vector3d color;      // Base color of the material
    double kd;           // Diffuse coefficient
    double ks;           // Specular coefficient
    double shine;        // Shininess exponent
    double transmittance; // Transmission coefficient for refraction
    double ior;          // Index of refraction
};

struct Fragment {
    Vector3d color;  // Fragment color
    double depth;    // Depth value of the fragment
};

struct Triangle {
    Vector3d v0, v1, v2; // Original vertices
    Vector3d tv0, tv1, tv2; // Transformed vertices
    Vector3d c0,c1,c2; //color/shade at each vertex
    Vector3d color; // Interpolated color
    Material material;
    Vector3d n0, n1, n2; // Normals
};



struct Polygon{
    std::vector<Vector3d> vertices;
    Material material;
};

struct PolyPatch{
    std::vector<Vector3d> normals;
    std::vector<Vector3d> vertices;
    Material material;
};

struct Light{
    Vector3d position; //xyz
    Vector3d color;
};

struct Camera{
    Vector3d up;
    Vector3d at;
    Vector3d from; 
    double hither;
    double aspect_ratio;
    double fov;
    int resX, resY; 
};

//the picture
struct Scene{ 
    Vector3d background; //background color
    Camera camera;
    std::vector<Triangle> triangles; //to store all triangle vector3d
    std::vector<Material> materials;
    std::vector<Polygon> polygons;
    std::vector<Light> lights;
    std::vector<PolyPatch> polyPatches;
    std::vector<std::vector<std::vector<Fragment>>> fragmentBuffer;
};

//**************all functions below*****************

/* parser stuff */
Vector3d vecParse(std::stringstream& ss){
    double x,y,z;
    ss >> x >> y >> z;
    return Vector3d(x,y,z);
}
void parser(const std::string& fileName, Scene& scene){
    std::ifstream file(fileName);
    if(!file.is_open()){
        std::cerr << fileName << " cannot be opened." << std::endl;
        return; 
    }

    std::string line; //define a line within a file
    Material currentMaterial; //tracks materials

    while(std::getline(file,line) && !line.empty()){
        std::stringstream ss(line);
        std::string current;
        ss >> current;   
        
        if(current == "b"){ //background color
            scene.background = vecParse(ss);
        }else if (current == "v"){ //view specification **borrowed from professor to fix an issue
            std::string junk;

            std::getline(file, line);
            std::stringstream fromss(line);
            fromss >> junk >> scene.camera.from[0] >> scene.camera.from[1] >> scene.camera.from[2];

            std::getline(file, line);
            std::stringstream atss(line);
            atss >> junk >> scene.camera.at[0] >> scene.camera.at[1] >> scene.camera.at[2];
    
            std::getline(file, line);
            std::stringstream upss(line);
            upss >> junk >> scene.camera.up[0] >> scene.camera.up[1] >> scene.camera.up[2];

            std::getline(file, line);
            std::stringstream angless(line);
            angless >> junk >> scene.camera.fov;

            std::getline(file, line);
            std::stringstream hitherss(line);
            hitherss >> junk >> scene.camera.hither;

            std::getline(file, line);
            std::stringstream resolutionss(line);
            resolutionss >> junk >> scene.camera.resX >> scene.camera.resY;
            
            scene.camera.aspect_ratio = static_cast<double>(scene.camera.resX) / scene.camera.resY;
        }
        else if(current == "l"){
            Light light;
            light.position = vecParse(ss);
            if (ss >> light.color[0] >> light.color[1] >> light.color[2]) {
                //Successfully parsed color values, no need to do anything
            }else{
                //No color values, set default color (white light)
                light.color = Vector3d(1.0, 1.0, 1.0);
            }
            scene.lights.push_back(light);
        }
        else if(current == "f"){ //material color
            double kd, ks, e, kt, ir; 
            currentMaterial.color = vecParse(ss); 
            ss >> kd >> ks >> e >> kt >> ir;
            currentMaterial.kd = kd;
            currentMaterial.ks = ks;
            currentMaterial.shine = e;
            currentMaterial.transmittance = kt;
            currentMaterial.ior = ir;
            scene.materials.push_back(currentMaterial);
            }
        else if(current == "p"){ //for polygons
            int numVerts;
            ss >> numVerts;
            if(numVerts == 3){ //cases for triangles
            Triangle tri;
            std::getline(file, line);
            std::stringstream v0Stream(line); 
            tri.v0 = vecParse(v0Stream);
            std::getline(file, line);
            std::stringstream v1Stream(line); 
            tri.v1 = vecParse(v1Stream);
            std::getline(file, line);
            std::stringstream v2Stream(line); 
            tri.v2= vecParse(v2Stream);
            tri.material = currentMaterial;
            scene.triangles.push_back(tri);
            }
            else if(numVerts == 4){ //quadrilaterals
                Triangle tri1, tri2; // split quad into two triangles
                Vector3d v0, v1, v2, v3;
                std::getline(file, line);
                std::stringstream v0Stream(line); 
                v0 = vecParse(v0Stream);
                std::getline(file, line);
                std::stringstream v1Stream(line); 
                v1 = vecParse(v1Stream);
                std::getline(file, line);
                std::stringstream v2Stream(line); 
                v2 = vecParse(v2Stream);
                std::getline(file, line);
                std::stringstream v3Stream(line); 
                v3 = vecParse(v3Stream);

                //Split quad into two triangles (v0, v1, v2) and (v2, v3, v0)
                tri1.v0 = v0; tri1.v1 = v1; tri1.v2 = v2;
                tri2.v0 = v2; tri2.v1 = v3; tri2.v2 = v0;

                Vector3d normal1 = (v1 - v0).cross(v2 - v0).normalized(); // Normal for tri1
                Vector3d normal2 = (v3 - v2).cross(v0 - v2).normalized(); // Normal for tri2

                // Assign normals to each vertex (for flat shading)
                tri1.n0 = tri1.n1 = tri1.n2 = normal1;
                tri2.n0 = tri2.n1 = tri2.n2 = normal2;

                tri1.material = tri2.material = currentMaterial;
                scene.triangles.push_back(tri1);
                scene.triangles.push_back(tri2);
            }else{ //for every other polygon 5 or more vertices
                 Polygon polygon;
            for (int i = 0; i < numVerts; ++i) {
                std::getline(file, line);  
                std::stringstream vertexStream(line);
                Vector3d vertex = vecParse(vertexStream);
                polygon.vertices.push_back(vertex);  
            }

            polygon.material = currentMaterial;  
            scene.polygons.push_back(polygon);  
            }
        }else if (current == "pp"){ //polygonal patches into triangles
            int numVerts;
            ss >> numVerts;
            PolyPatch polyPatch;
            for (int i = 0; i < numVerts; ++i) {
                std::getline(file, line);
                std::stringstream vertexStr(line);
                Vector3d vertex;
                Vector3d normal;
                vertexStr >> vertex[0] >> vertex[1] >> vertex[2] >> normal[0] >> normal[1] >> normal[2];
                polyPatch.vertices.push_back(vertex);
                polyPatch.normals.push_back(normal);
            }
            for (size_t i = 0; i < polyPatch.vertices.size() - 2; ++i) {
                Triangle tri;
                tri.v0 = polyPatch.vertices[0]; 
                tri.v1 = polyPatch.vertices[i + 1]; 
                tri.v2 = polyPatch.vertices[i + 2]; 

                tri.n0 = polyPatch.normals[0];
                tri.n1 = polyPatch.normals[i + 1];
                tri.n2 = polyPatch.normals[i + 2];

                tri.material = currentMaterial; 
                scene.triangles.push_back(tri); 
            }
        }  
    }
}


/*functions here*/


Vector3d shadeVertex(const Vector3d& position, const Vector3d& normal, const Material& material, Scene& scene,const Matrix4d& M ) {
    Vector3d localColor(0, 0, 0);
    Vector3d N = normal.normalized();
    Vector3d viewDir = (scene.camera.from - position).normalized(); // View direction

    // Number of lights
    double lightIntensity = 1.0 / sqrt(scene.lights.size());

    // Iterate over all lights
    for (const auto& light : scene.lights) {

        Vector3d L = (light.position - position).normalized(); // Light direction
        Vector3d H = (L + viewDir).normalized(); // Halfway vector

    
        // Compute diffuse and specular components
        double diffuse = std::max(0.0, N.dot(L));
        double specular = std::pow(std::max(0.0, N.dot(H)), material.shine);

        // Compute the color contribution from this light
        localColor[0] += (material.kd * material.color[0] * diffuse + material.ks * specular) * lightIntensity;
        localColor[1] += (material.kd * material.color[1] * diffuse + material.ks * specular) * lightIntensity;
        localColor[2] += (material.kd * material.color[2] * diffuse + material.ks * specular) * lightIntensity;
    }

    localColor = localColor.cwiseMin(Vector3d(1.0, 1.0, 1.0)).cwiseMax(Vector3d(0.0, 0.0, 0.0)); //clamp 

    return localColor;
}

void vertexProcessing(std::vector<Triangle>& triangles, const Matrix4d& M, Scene& scene) {
    for (Triangle& tri : triangles) {
        // Transform vertices
        Vector4d v0 = M * Vector4d(tri.v0.x(), tri.v0.y(), tri.v0.z(), 1.0);
        Vector4d v1 = M * Vector4d(tri.v1.x(), tri.v1.y(), tri.v1.z(), 1.0);
        Vector4d v2 = M * Vector4d(tri.v2.x(), tri.v2.y(), tri.v2.z(), 1.0);

        tri.tv0 = Vector3d(v0.x() / v0.w(), v0.y() / v0.w(), v0.z() / v0.w());
        tri.tv1 = Vector3d(v1.x() / v1.w(), v1.y() / v1.w(), v1.z() / v1.w());
        tri.tv2 = Vector3d(v2.x() / v2.w(), v2.y() / v2.w(), v2.z() / v2.w());

        // Compute color & light for each vertex
        tri.c0 = shadeVertex(tri.v0, tri.n0, tri.material, scene, M);
        tri.c1 = shadeVertex(tri.v1, tri.n1, tri.material, scene, M);
        tri.c2 = shadeVertex(tri.v2, tri.n2, tri.material, scene, M);
    }
}


Matrix4d computeViewMatrix(const Camera& camera) {
    Vector3d zaxis = (camera.from - camera.at).normalized();
    Vector3d xaxis = camera.up.cross(zaxis).normalized();
    Vector3d yaxis = zaxis.cross(xaxis);
    
    Matrix4d N_rotation;
    N_rotation << 
        xaxis(0),  xaxis(1),  xaxis(2),  0,
        yaxis(0),  yaxis(1),  yaxis(2),  0,
        zaxis(0),  zaxis(1),  zaxis(2),  0,
        0,         0,         0,         1; 

Matrix4d N_translation;
    N_translation << 
        1, 0, 0, -camera.from(0),
        0, 1, 0, -camera.from(1),
        0, 0, 1, -camera.from(2),
        0, 0, 0, 1;

    // Combined transformation matrix N
    Matrix4d viewMatrix = N_rotation * N_translation;
    return viewMatrix;
}

Matrix4d computePerspectiveMatrix(const Camera& camera) {
    double fov_rad = camera.fov * M_PI / 180.0;  // Convert to radians
    double near = camera.hither;
    double far = 1000.0 * camera.hither; 
    double top = tan(fov_rad / 2.0) * near;
    double bottom = -top;
    double right = top * camera.aspect_ratio;
    double left = -right;

    Matrix4d perspectiveMatrix;
    perspectiveMatrix << 
        -(2 * near) / (right - left), 0.0, (right + left) / (left - right), 0.0,
        0.0, -(2 * near) / (top - bottom), (top + bottom) / (bottom - top), 0.0,
        0.0, 0.0, (far + near) / (near - far), -(2 * far * near) / (far - near),
        0.0, 0.0, 1.0, 0.0;

    return perspectiveMatrix;
}

Matrix4d computeViewportMatrix(const Camera& camera) {
    Matrix4d viewportMatrix;
    viewportMatrix << camera.resX / 2.0, 0, 0, (camera.resX - 1.0) / 2.0,
                      0, camera.resY / 2.0, 0, (camera.resY - 1.0) / 2.0,
                      0, 0, 1, 0,
                      0, 0, 0, 1;
    return viewportMatrix;
}

Matrix4d computeCombinedMatrix(const Camera& camera) {
    Matrix4d M_vp = computeViewportMatrix(camera);
    Matrix4d M_per = computePerspectiveMatrix(camera);
    Matrix4d M_cam = computeViewMatrix(camera);
    printMatrix(M_vp, "M_vp");
    printMatrix(M_per, "M_per");
    printMatrix(M_cam, "M_cam");

    return M_vp * M_per * M_cam; // Combined transformation
}

Matrix4d createModelMatrix() {
    // Simply return the identity matrix
    return Matrix4d::Identity();
}



//render helper functions

Vector3d computeBarycentricCoordinates(const Triangle& tri, double x, double y) {
    // Triangle vertices
    Vector3d v0 = tri.tv0;
    Vector3d v1 = tri.tv1;
    Vector3d v2 = tri.tv2;

    // 2x2 matrix for the linear system
    Eigen::Matrix2d A;
    A << v1.x() - v0.x(), v2.x() - v0.x(),
         v1.y() - v0.y(), v2.y() - v0.y();
    
    // Right-hand side vector
    Eigen::Vector2d rhs(x - v0.x(), y - v0.y());

    // Solve for beta and gamma
    Eigen::Vector2d result = A.inverse() * rhs;
    double beta = result(0);
    double gamma = result(1);

    // Calculate alpha based on the constraint: alpha + beta + gamma = 1
    double alpha = 1.0 - beta - gamma;

    return Vector3d(alpha, beta, gamma);
}


bool isInsideTriangle(const Vector3d& bary) {
    return bary[0] >= 0 && bary[1] >= 0 && bary[2] >= 0;
}


Vector3d fragmentShading(const Vector3d& baseColor, const Vector3d& normal, const Material& material, const std::vector<Light>& lights, const Camera& camera) {
    // Initialize the final color
    Vector3d finalColor(0, 0, 0);
    double lightIntensity = 1.0 / sqrt(lights.size());
    // Camera position (for specular reflection)
    Vector3d viewDir = (camera.from - normal).normalized();

    for (const auto& light : lights) {

        // Diffuse lighting 
        Vector3d lightDir = (light.position - normal).normalized();
        double diffuseFactor = std::max(0.0, normal.dot(lightDir));

        //Specular lighting 
        Vector3d reflectDir = (lightDir + viewDir).normalized();
        double specFactor = std::pow(std::max(0.0, normal.dot(reflectDir)), material.shine);

        // Combine all components
        finalColor[0] += (material.kd * material.color[0] * diffuseFactor + material.ks * specFactor) * lightIntensity;  
        finalColor[1] += (material.kd * material.color[1] * diffuseFactor + material.ks * specFactor) * lightIntensity;
        finalColor[2] += (material.kd * material.color[2] * diffuseFactor + material.ks * specFactor) * lightIntensity;

        finalColor = finalColor.cwiseMin(Vector3d(1.0, 1.0, 1.0)).cwiseMax(Vector3d(0.0, 0.0, 0.0)); //clamp
    }

    return finalColor;
}


Vector3d interpolateColor(const Vector3d& c0, const Vector3d& c1, const Vector3d& c2, const Vector3d& barycentricCoords) {
    return barycentricCoords[0] * c0 + barycentricCoords[1] * c1 + barycentricCoords[2] * c2;
}

double interpolateDepth(const Triangle& tri, const Vector3d& barycentricCoords) {
    return barycentricCoords[0] * tri.tv0.z() + barycentricCoords[1] * tri.tv1.z() + barycentricCoords[2] * tri.tv2.z();
}

void rasterizeTriangle(const Triangle& tri, unsigned char* framebuffer, DepthBuffer& depthBuffer, int width, int height, Scene& scene, bool fragment) {
    // Compute the 2D bounding box of the triangle
    double minX = std::min({tri.tv0.x(), tri.tv1.x(), tri.tv2.x()});
    double maxX = std::max({tri.tv0.x(), tri.tv1.x(), tri.tv2.x()});
    double minY = std::min({tri.tv0.y(), tri.tv1.y(), tri.tv2.y()});
    double maxY = std::max({tri.tv0.y(), tri.tv1.y(), tri.tv2.y()});
    
    // Clamp the bounding box to the screen size
    int xMin = std::max(0, static_cast<int>(std::floor(minX)));
    int xMax = std::min(width - 1, static_cast<int>(std::ceil(maxX)));
    int yMin = std::max(0, static_cast<int>(std::floor(minY)));
    int yMax = std::min(height - 1, static_cast<int>(std::ceil(maxY)));
    
    double w0 = 1.0 / tri.tv0.z();
    double w1 = 1.0 / tri.tv1.z();
    double w2 = 1.0 / tri.tv2.z();

    // Ensure the fragment buffer is the right size
    if (scene.fragmentBuffer.size() != height || scene.fragmentBuffer[0].size() != width) {
        scene.fragmentBuffer.resize(height, std::vector<std::vector<Fragment>>(width));
    }

    // Loop over the pixels in the bounding box
    for (int y = yMin; y <= yMax; ++y) {
        for (int x = xMin; x <= xMax; ++x) {
            // Compute barycentric coordinates for the current pixel
            Vector3d bary = computeBarycentricCoordinates(tri, x, y);
            
            // If the pixel is inside the triangle (all barycentric coords >= 0)
            if (isInsideTriangle(bary)) {
                // Interpolate depth
                double depth = bary[0] * w0 + bary[1] * w1 + bary[2] * w2;
                
                // Depth test using DepthBuffer class
                if (depth < depthBuffer.get(x, y)) {
                    depthBuffer.set(x, y, depth);
                    
                    // Interpolate color based on vertex colors
                    Vector3d interpolatedColor(0, 0, 0);
                    Fragment newFragment;

                    if (fragment) {
                        Vector3d interpolatedNormal = (bary[0] * tri.n0 / w0 + bary[1] * tri.n1 / w1 + bary[2] * tri.n2 / w2) /
                                                       (bary[0] / w0 + bary[1] / w1 + bary[2] / w2);
                        interpolatedNormal.normalize();  // Normalize the interpolated normal

                        // Perspective-correct color interpolation
                        interpolatedColor = (bary[0] * tri.c0 / w0 + bary[1] * tri.c1 / w1 + bary[2] * tri.c2 / w2) /
                                            (bary[0] / w0 + bary[1] / w1 + bary[2] / w2);

                        //fragment shading
                        Vector3d shadedColor = fragmentShading(interpolatedColor, interpolatedNormal, tri.material, scene.lights, scene.camera);
                        newFragment = {shadedColor, depth};
                    } else {
                        //flat shading
                        interpolatedColor = bary[0] * tri.c0 + bary[1] * tri.c1 + bary[2] * tri.c2;
                        newFragment = {interpolatedColor, depth};
                    }

                    // Store the new fragment 
                    scene.fragmentBuffer[y][x].push_back(newFragment);
                }
            }
        }
    }
}


void blend(unsigned char* framebuffer, const std::vector<std::vector<std::vector<Fragment>>>& fragmentBuffer, int width, int height, float trans) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Collect fragments for the current pixel
            const auto& fragments = fragmentBuffer[y][x];
            if (!fragments.empty()) {
                // Copy fragments for sorting
                std::vector<Fragment> sortedFragments = fragments;

                // Sort fragments by depth in reverse order (farthest to nearest)
                std::sort(sortedFragments.begin(), sortedFragments.end(), [](const Fragment& a, const Fragment& b) {
                    return a.depth > b.depth; // Sort in descending order
                });

                // Retrieve the background color from the framebuffer
                int pixelIdx = ((height - y - 1) * width + x) * 3;
                Vector3d currentColor;
                currentColor[0] = framebuffer[pixelIdx] / 255.0f;
                currentColor[1] = framebuffer[pixelIdx + 1] / 255.0f;
                currentColor[2] = framebuffer[pixelIdx + 2] / 255.0f;

                // Loop through each sorted fragment at this pixel position
                for (const auto& fragment : sortedFragments) {
                    // Use the trans value as alpha 
                    float alpha = trans;

                    // Blend each fragment's color with the current color
                    currentColor = alpha * fragment.color + (1.0f - alpha) * currentColor;
                }

                // Clamp the blended color to the [0, 1] range
                currentColor = currentColor.cwiseMin(1.0).cwiseMax(0.0);

                // Set the final blended color to the framebuffer
                framebuffer[pixelIdx] = static_cast<unsigned char>(currentColor[0] * 255);
                framebuffer[pixelIdx + 1] = static_cast<unsigned char>(currentColor[1] * 255);
                framebuffer[pixelIdx + 2] = static_cast<unsigned char>(currentColor[2] * 255);
            }
        }
    }
}
void rasterizeScene(const std::vector<Triangle>& triangles, unsigned char* framebuffer, DepthBuffer& depthBuffer, int width, int height, 
                                        double trans, const Matrix4d& M, Scene& scene, bool fragment) {

    for (const Triangle& tri : triangles) {
        rasterizeTriangle(tri, framebuffer, depthBuffer, width, height, scene, fragment);
    }

    // Blend the fragment buffer with the framebuffer
    blend(framebuffer, scene.fragmentBuffer, width, height, trans);
}

/*
bool clipTriangleAgainstFrustum(const Triangle& triangle, std::vector<Triangle>& clippedTriangles) {
    // Define the six frustum planes in view space (normalized to have a unit length normal)
    // Left, Right, Bottom, Top, Near, Far planes are represented by a point and a normal vector
    std::vector<Vector4d> frustumPlanes = {
        Vector4d( 1,  0,  0, 1), // Left plane
        Vector4d(-1,  0,  0, 1), // Right plane
        Vector4d( 0,  1,  0, 1), // Bottom plane
        Vector4d( 0, -1,  0, 1), // Top plane
        Vector4d( 0,  0,  1, 1), // Near plane
        Vector4d( 0,  0, -1, 1000) // Far plane (assuming far = 1000)
    };

    // Check if all vertices are outside of any frustum plane
    for (const Vector4d& plane : frustumPlanes) {
        int outsideCount = 0;

        // Check each vertex of the triangle (tv0, tv1, tv2) against the current plane
        if (plane.dot(Vector4d(triangle.tv0[0], triangle.tv0[1], triangle.tv0[2], 1.0)) < 0) outsideCount++;
        if (plane.dot(Vector4d(triangle.tv1[0], triangle.tv1[1], triangle.tv1[2], 1.0)) < 0) outsideCount++;
        if (plane.dot(Vector4d(triangle.tv2[0], triangle.tv2[1], triangle.tv2[2], 1.0)) < 0) outsideCount++;

        // If all vertices are outside of the plane, discard the triangle
        if (outsideCount == 3) {
            return false;
        }
    }

    // If some part of the triangle is inside the frustum, add it to the clippedTriangles vector
    clippedTriangles.push_back(triangle);

    return true;
}
*/


//output file format
void saveImage(const std::string& outputName, unsigned char* pixels, int width, int height) {
    // Open file in binary mode
    std::ofstream outFile(outputName, std::ios::binary);
    
    if (!outFile) {
        std::cerr << "Error: Could not open file " << outputName << " for writing." << std::endl;
        return;
    }
    
    // Write the PPM header
    outFile << "P6\n" << width << " " << height << "\n255\n";
    
    // Write pixel data in binary format (RGB for each pixel)
    outFile.write(reinterpret_cast<char*>(pixels), width * height * 3);
    
    // Close the file
    outFile.close();

    if (outFile.fail()) {
        std::cerr << "Error: Could not write image data to file " << outputName << std::endl;
    } else {
        std::cout << "Image saved successfully as " << outputName << std::endl;
    }
}
//rasterize render
void render(Scene& scene, const std::string& outputName, float trans, bool fragmentShading) {
    unsigned char* pixels = new unsigned char[scene.camera.resX * scene.camera.resY * 3];
    DepthBuffer depthBuffer(scene.camera.resX, scene.camera.resY);
    
     // Initialize pixels to the background color from the scene
    for (int i = 0; i < scene.camera.resX * scene.camera.resY; ++i) {
        pixels[i * 3] = static_cast<unsigned char>(scene.background[0] * 255); // Red
        pixels[i * 3 + 1] = static_cast<unsigned char>(scene.background[1] * 255); // Green
        pixels[i * 3 + 2] = static_cast<unsigned char>(scene.background[2] * 255); // Blue
    }
    
    depthBuffer.clear(); // set all depths to max

    Matrix4d M_model = createModelMatrix();
    Matrix4d M = computeCombinedMatrix(scene.camera) * M_model;

    printMatrix(M, "Combined Matrix");

    vertexProcessing(scene.triangles, M, scene);
    
    rasterizeScene(scene.triangles, pixels, depthBuffer, scene.camera.resX, scene.camera.resY, trans, M, scene, fragmentShading);

    saveImage(outputName, pixels, scene.camera.resX, scene.camera.resY);
    
    delete[] pixels;
}


//print out information to see if parser worked correctly
void printCameraInfo(const Camera& camera) {
    std::cout << "Camera Information:" << std::endl;
    std::cout << "From: " << camera.from.transpose() << std::endl;
    std::cout << "At: " << camera.at.transpose() << std::endl;
    std::cout << "Up: " << camera.up.transpose() << std::endl;
    std::cout << "Field of View (FOV): " << camera.fov << std::endl;
    std::cout << "Hither (near plane): " << camera.hither << std::endl;
    std::cout << "Resolution: " << camera.resX << "x" << camera.resY << std::endl;
    std::cout << "Aspect Ratio: " << camera.aspect_ratio << std::endl;

}


int main(int argc, char** argv){
    srand(static_cast<unsigned int>(time(nullptr))); // Seed random number generator
    float transparencyVal = 1.0f; //default transparency value
    bool shadowMap = false; //not implemented
    bool fragmentShade = false;
    int opt;

     while ((opt = getopt(argc, argv, "t:fj")) != -1) {
        switch (opt) {
            case 't':
                transparencyVal = std::stof(optarg); 
                break;
            case 'f':
                fragmentShade = true;
                break;
            case 's':
                shadowMap = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-s sqrt of total samples]" << std::endl;
                exit(EXIT_FAILURE);
        }
    }
        if (optind + 1 >= argc) { 
        std::cerr << "Error: Expected input and output filenames after options." << std::endl;
        return EXIT_FAILURE;
    }
      const char* nffFile = argv[optind];
      const char* outputFile = argv[optind + 1];


    if (shadowMap) {
        std::cout << "Shadow Map is enabled." << std::endl;
    } else {
        std::cout << "Shadow Map is disabled." << std::endl;
    }
    if (fragmentShade) {
        std::cout << "Fragment shading is enabled." << std::endl;
    } else {
        std::cout << "Fragment shading is disabled." << std::endl;
    }
    Scene scene;
    parser(nffFile, scene);
    printCameraInfo(scene.camera);
    std::cout << "Background Color: " << scene.background.transpose() << std::endl;
    for(const auto& light:scene.lights){
     std::cout << "Light: (" << light.position[0] << ", " << light.position[1] << ", " << light.position[2] << ")" << std::endl;
}

    

/*
    for (Triangle& triangle : scene.triangles) {
        std::cout << "Transformed Triangle Vertices:\n";
    std::cout << triangle.tv0.transpose() << "\n";
    std::cout << triangle.tv1.transpose() << "\n";
    std::cout << triangle.tv2.transpose() << "\n";
    }
*/

    std::cout << "------------------------------------" << std::endl; // Separator for readability
    std::cout << "Number of triangles parsed: " << scene.triangles.size() << std::endl;
    std::cout << "Number of polygons parsed: " << scene.polygons.size() << std::endl;
    std::cout << "Number of polygon patches parsed: " << scene.polyPatches.size() << std::endl;
    std::cout << "Number or lights parsed: " << scene.lights.size() << std::endl;
    std::cout << "Transparency Value: " << transparencyVal << std::endl;
    

    auto start = std::chrono::high_resolution_clock::now();

    render(scene, outputFile, transparencyVal, fragmentShade); //THE RENDER

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nRender time: " << elapsed.count() << " seconds" << std::endl;
    std::cout<<"\nDone!"<<std::endl;


    return 0;

}
