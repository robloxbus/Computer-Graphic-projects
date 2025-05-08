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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Eigen::Vector3d;
using Eigen::Vector2d; 

/*OVERLOADERS*/

std::ostream& operator<<(std::ostream& os, const Vector3d& vec) {
    os << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
    return os;
}

//*************all classes/structures below**********************

//to keep track of multiple material color values
struct Material {
    Vector3d color;      // Base color of the material
    double kd;           // Diffuse coefficient
    double ks;           // Specular coefficient
    double shine;        // Shininess exponent
    double transmittance; // Transmission coefficient for refraction
    double ior;          // Index of refraction
    double T;           //transmittance
    bool Reflective; 
    bool Transparent;
};

enum HitType { TRIANGLE, POLYGON, SPHERE, UNKNOWN };

struct Triangle{
    Vector3d v0, v1, v2;
    Material material;
    Vector3d n0,n1,n2;
    Vector3d normal;
};

struct Sphere{
    Vector3d center;
    double radius;
    Material material;
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

struct hitRecord{
    Vector3d intPoint; //point of intersection
    Material material;
    Vector3d normal; //surface normal at intersection
    double t; //the distance from origin to intersection point
    bool intersection; 
    Vector3d v; //view vector
    Vector3d n0, n1, n2; //normals at triangle vertices
    double a, b, g; //alpha, beta, gamma
    HitType type;
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
    std::vector<Sphere> spheres;
    std::vector<Material> materials;
    std::vector<Polygon> polygons;
    std::vector<Light> lights;
    std::vector<PolyPatch> polyPatches;
};

struct Ray{
    Vector3d origin;
    Vector3d direction;
    int depth;
};

//**************all functions below*****************

//a parser specifically for parsing stuff into a vector3d
Vector3d vecParse(std::stringstream& ss){
    double x,y,z;
    ss >> x >> y >> z;
    return Vector3d(x,y,z);
}


//The parser that parses the file that needs the parsing
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
        }else if(current == "s"){ //round object
            Sphere sphere;
            ss >> sphere.center[0] >> sphere.center[1] >> sphere.center[2] >> sphere.radius;
            sphere.material = currentMaterial;
            scene.spheres.push_back(sphere); 
        }else if (current == "pp"){ //polygonal patches
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
            polyPatch.material = currentMaterial;
            scene.polyPatches.push_back(polyPatch); 
        }
    }
}

Ray makeRay(const Camera& camera, double pixelX, double pixelY, double apSize){
    Vector3d w = (camera.from - camera.at).normalized(); //forward vector
    Vector3d u = camera.up.cross(w).normalized(); //right vector
    Vector3d v = w.cross(u).normalized(); //up vector

    double focalDistance = (camera.from - camera.at).norm(); //Use the distance to the focal point

    //Calculate image plane size at the focal distance
    double fovRad = (camera.fov * M_PI) / 180.0;
    double aspect_ratio = camera.aspect_ratio;

    double height = 2.0 * tan(fovRad / 2.0) * focalDistance;
    double width = height * aspect_ratio;

    //Pixel size in world space
    double pixelWidth = width / camera.resX;
    double pixelHeight = height / camera.resY;

    //Calculate pixel position on the image plane
    double halfWidth = width / 2.0;
    double halfHeight = height / 2.0;

    //Pixel position on the image plane
    double a = (pixelX + 0.5) * pixelWidth - halfWidth;
    double b = (pixelY + 0.5) * pixelHeight - halfHeight;

    Ray ray;

    //Calculate the center ray direction towards the focal plane
    Vector3d direction = (a * u + b * v - focalDistance * w).normalized();

    if (apSize > 0.0) {
        //Generate a random offset within the lens aperture
        double randomX = ((double)rand() / RAND_MAX - 0.5) * apSize;
        double randomY = ((double)rand() / RAND_MAX - 0.5) * apSize;

        //Calculate lens point based on aperture size
        Vector3d lensPoint = camera.from + (randomX * u) + (randomY * v);
        
        //Update ray origin to the lens point
        ray.origin = lensPoint;
         Vector3d focalPoint = camera.from - focalDistance * w + (a * u) + (b * v);

        //Calculate the direction from the lens point to the focal point
        ray.direction = (focalPoint - lensPoint).normalized(); //Aim at the focal point ; from - at
    } else {
    
        ray.origin = camera.from; 
        ray.direction = direction; 
    }

    return ray;
}


bool hitRecordTrack(const Ray& ray, const Triangle& triangle, hitRecord& hit){
    const double EPSILON = 1e-8; //Small value to handle precision errors

    Vector3d v0 = triangle.v0;
    Vector3d v1 = triangle.v1;
    Vector3d v2 = triangle.v2;

    Vector3d edge1 = v1 - v0;
    Vector3d edge2 = v2 - v0;
    
    Vector3d h = ray.direction.cross(edge2); //h is perpendicular to edge2 and ray direction
    double a = edge1.dot(h); //a is dot product between edge 1 and h

    //If a is close to zero, the ray is parallel to the triangle, no intersection
    if (a > -EPSILON && a < EPSILON) {
        return false;
    }

    double f = 1.0 / a; //inverse of 'a' to scale 
    Vector3d s = ray.origin - v0;
    double alpha = f * s.dot(h);  //alpha is the first barycentric coordinate

    //Check if alpha is within the valid range
    if (alpha < 0.0 || alpha > 1.0) {
        return false;
    }

    Vector3d q = s.cross(edge1);
    double beta = f * ray.direction.dot(q);  //beta is the second barycentric coordinate

    //Check if beta is within the valid range and if alpha + beta <= 1.0
    if (beta < 0.0 || alpha + beta > 1.0) {
        return false;
    }

    //at this point, ray intersects triangle; compute t to find intersection point
    double t = f * edge2.dot(q);

    //Ray intersection happens if t > EPSILON 
    if (t > EPSILON) {
        hit.intPoint = ray.origin + t * ray.direction; // Intersection point
        hit.t = t;
        hit.normal = edge1.cross(edge2).normalized(); // Normal of triangle
        hit.a = alpha; //barycentric coordinates
        hit.b = beta;
        hit.g = 1.0 - alpha - beta; // gamma
        hit.material = triangle.material; // Assign material
        hit.intersection = true; //Mark as an intersection
        hit.v = (ray.origin - hit.intPoint).normalized(); //View vector
        hit.type = TRIANGLE;
        return true;
    }

    //If t is not greater than EPSILON, there's no intersection
    return false;

}


//Function to check if a ray intersects with a sphere
bool hitRecordSphere(const Ray& ray, const Sphere& sphere, hitRecord& rec) {
    //Compute the vector from the ray's origin to the center of the sphere
    Vector3d e_c = ray.origin - sphere.center;

    //Calculate coefficients for the quadratic equation 
    double a = ray.direction.dot(ray.direction); 
    double b = 2.0 * e_c.dot(ray.direction);     
    double c = e_c.dot(e_c) - sphere.radius * sphere.radius; 

    //Compute the discriminant (b^2 - 4ac) 
    double discriminant = (b * b) - (4 * a * c);
    if (discriminant < 0){
        return false; 
    }  

    //solve for intersection points
    double sqrtDiscriminant = sqrt(discriminant);
    double t1 = (-b - sqrtDiscriminant) / (2.0 * a); 
    double t2 = (-b + sqrtDiscriminant) / (2.0 * a); 

    //Choose the smallest positive t value aka the closest intersection point
    double t = (t1 > 0) ? t1 : t2;  
    if (t < 0){
         return false; 
    }
    //Record the hit information 
    rec.t = t;
    rec.intPoint = ray.origin + t * ray.direction; 
    rec.material = sphere.material; 
    rec.normal = (rec.intPoint - sphere.center).normalized();
    rec.v = (ray.origin - rec.intPoint).normalized();           
    rec.intersection = true;                  

    return true; 
}

// Function to calculate the intersection point of a ray with a plane defined by a polygon.
Vector3d intersectRayWithPlane(const Ray& ray, const Polygon& polygon) {
    //Calculate the normal of the polygon (plane) using two edges of the polygon
    Vector3d normal = (polygon.vertices[1] - polygon.vertices[0]).cross(polygon.vertices[2] - polygon.vertices[0]);
    normal.normalize();  //Normalize the normal vector to get a unit vector
    
    //Compute the dot product of the ray direction and the normal of the plane
    double denom = normal.dot(ray.direction.normalized());
    
    //Calculate the t value for the parametric equation; this t gives us the distance along the ray to the plane
    double t = (polygon.vertices[0] - ray.origin).dot(normal) / denom;

    //Return the intersection point using the parametric equation
    return ray.origin + t * ray.direction;
}

//Function to project a 3D point onto a 2D plane defined by the polygon
Vector2d projectTo2D(const Vector3d& point, const Polygon& polygon) {

    Vector2d projectedPoint; //Return 2D vector

    Vector3d v0 = polygon.vertices[0];
    Vector3d v1 = polygon.vertices[1];
    Vector3d v2 = polygon.vertices[2];
    
    Vector3d edge1 = v1 - v0;
    Vector3d edge2 = v2 - v0;
    
    //Calculate the 3D vector from v0 to the point
    Vector3d p = point - v0;
    
    //Project the 3D point onto the 2D plane defined by the two edges
    projectedPoint.x() = p.dot(edge1);
    projectedPoint.y() = p.dot(edge2);
    
    return projectedPoint;
}

//Function to determine if a 2D point lies within a polygon using the test-ray method and  counts intersections with polygon edges and checks if the count is odd (inside) or even (outside).
bool isPointInPolygon2D(const Vector2d& point, const std::vector<Vector2d>& polygon2D) {

    int numIntersections = 0;   //Count of ray-polygon edge intersections
    
    //Loop through each edge of the polygon
    for (int i = 0; i < polygon2D.size(); ++i) {
        //Get the current and next vertex of the polygon 
        Vector2d v0 = polygon2D[i];
        Vector2d v1 = polygon2D[(i + 1) % polygon2D.size()];
        
        //Check if the ray intersects this edge of the polygon
        if ((v0.y() > point.y()) != (v1.y() > point.y())) {
            //Compute the X-coordinate of the intersection point between the ray and the polygon edge
            double xIntersection = (v1.x() - v0.x()) * (point.y() - v0.y()) / (v1.y() - v0.y()) + v0.x();
            
            //If X coordinate of intersection is right of point, count as intersection
            if (point.x() < xIntersection) {
                numIntersections++;
            }
        }
    }
    //Return true if the number of intersections is odd, false if even
    return (numIntersections % 2 == 1);
}

bool hitRecordTrackConcave(const Ray& ray, const Polygon& polygon, hitRecord& HitRecord) {
    //Compute the intersection with the polygon plane
    Vector3d intersectionPoint = intersectRayWithPlane(ray, polygon);

    //Project the intersection point to 2D
    Vector2d projectedPoint = projectTo2D(intersectionPoint, polygon);

    //Convert polygon vertices to 2D for the point-in-polygon test
    std::vector<Vector2d> polygon2D;
    for (const Vector3d& vertex : polygon.vertices) {
        polygon2D.push_back(projectTo2D(vertex, polygon));
        }

    //Check if the projected point is inside the 2D polygon
    if (isPointInPolygon2D(projectedPoint, polygon2D)) {
        HitRecord.intPoint = intersectionPoint;
        HitRecord.t = (intersectionPoint - ray.origin).norm();
        HitRecord.material = polygon.material;
        HitRecord.intersection = true;
        Vector3d v0 = polygon.vertices[0];
        Vector3d v1 = polygon.vertices[1];
        Vector3d v2 = polygon.vertices[2]; // Assumes the polygon is defined with at least 3 vertices
        Vector3d edge1 = v1 - v0;
        Vector3d edge2 = v2 - v0;
        HitRecord.normal = edge1.cross(edge2).normalized();
        HitRecord.v = (ray.origin - HitRecord.intPoint).normalized();
        return true;
        }
    return false;
}

bool polyPatchHitRecord(const Ray& ray, const PolyPatch& polyPatch, hitRecord& hit){
    bool foundHit = false;
    hitRecord tempHit;
    tempHit.t = std::numeric_limits<double>::max(); // Start with the maximum possible distance

    // Iterate through the polygon vertices to form triangles
    for (size_t i = 1; i < polyPatch.vertices.size() - 1; ++i) {
        Triangle triangle;
        triangle.v0 = polyPatch.vertices[0]; // Fan base vertex
        triangle.v1 = polyPatch.vertices[i];   // Second vertex
        triangle.v2 = polyPatch.vertices[i + 1]; // Third vertex
        triangle.n0 = polyPatch.normals[0];
        triangle.n1 = polyPatch.normals[i];
        triangle.n2 = polyPatch.normals[i + 1];
        triangle.material = polyPatch.material; // Assign the material

        // Check for intersection
        hitRecord tempTriangleHit;
        if (hitRecordTrack(ray, triangle, tempTriangleHit)) {
            if (tempTriangleHit.t < tempHit.t) {
                tempHit = tempTriangleHit; 
                foundHit = true; 
                tempHit.n0 = triangle.n0; // Store normals
                tempHit.n1 = triangle.n1;
                tempHit.n2 = triangle.n2;
            }
        }
    }
    if (foundHit) {
        hit = tempHit;
        hit.material = polyPatch.material; 
        hit.normal = (tempHit.normal).normalized();
        hit.v = (ray.origin - hit.intPoint).normalized(); 
        hit.type = POLYGON;
    }
    hit.intersection = foundHit; 
    return foundHit;
}


Vector3d computeLighting(const hitRecord& hit, const Scene& scene, const Ray& ray, bool phong) {
    Vector3d localColor(0, 0, 0);
    Vector3d N; 

    if(phong && (hit.type != TRIANGLE)){
         N = (hit.a * hit.n1 + hit.b * hit.n2 + hit.g * hit.n0).normalized();
    }else{
        N = hit.normal.normalized();
    }
    Vector3d P = hit.intPoint; // Intersection point
    Vector3d viewDir = (ray.origin - P).normalized(); // View direction

    // Number of lights
    double lightIntensity = 1.0 / sqrt(scene.lights.size());

    // Iterate over all lights
    for (const auto& light : scene.lights) {
        Vector3d L = (light.position - P).normalized(); // Light direction
        Vector3d H = (L + viewDir).normalized(); // Halfway vector

        // Initialize shadow ray
        Ray shadowRay;
        shadowRay.origin = P + N * 1e-4;
        shadowRay.direction = L;

        bool inShadow = false;
        hitRecord shadowHit;
        // Check for shadows
        for (const auto& triangle : scene.triangles) {
            if (hitRecordTrack(shadowRay, triangle, shadowHit) && shadowHit.t > 0 && shadowHit.t < (light.position - P).norm()) {
                inShadow = true;
                break;
            }
        }

        for (const auto& sphere : scene.spheres) {
            if (hitRecordSphere(shadowRay, sphere, shadowHit) && shadowHit.t > 0 && shadowHit.t < (light.position - P).norm()) {
                inShadow = true;
                break;
            }
        }

         for (const auto& polygon : scene.polygons) {
                if (hitRecordTrackConcave(shadowRay, polygon, shadowHit) && shadowHit.t < (light.position - P).norm()) {
                    inShadow = true;
                    break;
                }
            }
            for (const auto& polyPatch : scene.polyPatches) {
                if (polyPatchHitRecord(shadowRay, polyPatch, shadowHit) && shadowHit.t < (light.position - P).norm()) {
                    inShadow = true;
                    break;
                }
            }

        if (inShadow) continue; //if shadow ray found

        // Compute diffuse and specular 
        double diffuse = std::max(0.0, N.dot(L));
        double specular = std::pow(std::max(0.0, N.dot(H)), hit.material.shine);

        //Compute the color contribution from this light
        localColor[0] += (hit.material.kd * hit.material.color[0] * diffuse + hit.material.ks * specular) * lightIntensity;
        localColor[1] += (hit.material.kd * hit.material.color[1] * diffuse + hit.material.ks * specular) * lightIntensity;
        localColor[2] += (hit.material.kd * hit.material.color[2] * diffuse + hit.material.ks * specular) * lightIntensity;


    }
    return localColor;
}



//*****tracing and rendering below *******************/


//returns the color for each pixel while keeping hit record in mind
Vector3d traceRay(const Ray &ray, const Scene& scene, bool phong, int depth=0){
    hitRecord closestHit;
    closestHit.intersection = false;
    closestHit.t = std::numeric_limits<double>::max();

    //for triangles.
    for(const auto& triangle: scene.triangles){
        hitRecord temp;
        if(hitRecordTrack(ray, triangle, temp) && temp.t < closestHit.t){
            closestHit = temp;
        }
    }
    //round object
    for (const auto& sphere : scene.spheres){
        hitRecord temp2; 
        if(hitRecordSphere(ray, sphere, temp2) && temp2.t < closestHit.t){
            closestHit = temp2;
        }

    }
    //for cases concerning polygons (5 or more vertices)
     for(const auto& polygon : scene.polygons) {
        hitRecord temp;
        if (hitRecordTrackConcave(ray, polygon, temp) && temp.t < closestHit.t) {
            closestHit = temp;
        }
    }

    for(const auto& polypatch : scene.polyPatches){
        hitRecord temp;
        if (polyPatchHitRecord(ray, polypatch, temp) && temp.t < closestHit.t){
            closestHit = temp;
        }
    }

    if(closestHit.intersection){

        Vector3d localColor(0, 0, 0);
        Vector3d N = closestHit.normal.normalized(); // Get the normal
        Vector3d P = closestHit.intPoint; // Intersection point
        Vector3d viewDir = (ray.origin - P).normalized(); // View direction
        
        localColor += computeLighting(closestHit, scene, ray, phong); //adds shadow, colors, the good stuff

        //Add reflection
        if (closestHit.material.ks > 0 && depth < 5) {
            Vector3d reflectedDir = (ray.direction - 2 * closestHit.normal.dot(ray.direction) * closestHit.normal);
            Ray reflectedRay;
            reflectedRay.origin = closestHit.intPoint;
            reflectedRay.direction = reflectedDir;
            reflectedRay.depth = depth + 1;
            localColor += closestHit.material.ks * traceRay(reflectedRay, scene, phong, depth + 1);
        }

        localColor = localColor.cwiseMin(1.0).cwiseMax(0.0); //to make sure colors are clamped so no neon colors
 
        return localColor;
    }else {
        return scene.background;
    }
}

//output file format

void render(const Scene& scene, bool jittering, int numSamples, double apSize, bool phongEnabled, const std::string& outputName){
    unsigned char *pixels = new unsigned char[scene.camera.resX * scene.camera.resY * 3];
    int totalSamples = numSamples * numSamples; // Number of samples is used for stratified sampling
    double pixelW = 1.0/numSamples;
    double pixelH = 1.0/numSamples;
    for (int i = 0; i < scene.camera.resY; ++i){
        for (int j = 0; j < scene.camera.resX; ++j){
            Vector3d color(0, 0, 0); // Initialize color for the pixel

            if(jittering){

                for (int sampleY = 0; sampleY < numSamples; ++sampleY) {
                    for (int sampleX = 0; sampleX < numSamples; ++sampleX) {

                        double jitterX = ((double)rand() / RAND_MAX) * pixelW;
                        double jitterY = ((double)rand() / RAND_MAX) * pixelH;

                        // Calculate the offset within the pixel
                        double offsetX = j + (sampleX + jitterX) * pixelW;
                        double offsetY = i + (sampleY + jitterY) * pixelH;
                        Ray ray = makeRay(scene.camera, offsetX, offsetY, apSize);
                        color += traceRay(ray, scene, phongEnabled);
                    }
                }
                //Average the accumulated color over the total number of samples
                color /= totalSamples;
            }else{
                Ray ray = makeRay(scene.camera, j, i, apSize);
                color += traceRay(ray, scene, phongEnabled);
            }
            int index = ((scene.camera.resY - 1 - i) * scene.camera.resX + j) * 3;
            pixels[index] = static_cast<unsigned char>(color[0] * 255); // R
            pixels[index + 1] = static_cast<unsigned char>(color[1] * 255); // G
            pixels[index + 2] = static_cast<unsigned char>(color[2] * 255); // B

        }
        //progress tracker: calculate percentage done after completing each row
        double progress = static_cast<double>(i + 1) / scene.camera.resY * 100;
        std::cout << "\rRendering progress: " << static_cast<int>(progress) << "% completed." << std::flush;
    }
    std::ofstream out(outputName, std::ios::out | std::ios::binary);
    out << "P6\n" << scene.camera.resX << " " << scene.camera.resY << "\n255\n";
    out.write(reinterpret_cast<char*>(pixels), scene.camera.resX * scene.camera.resY * 3);
    out.close();

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
    int numSamples = 1;
    double apertureSize = 0.0;
    bool jitteringEnabled = false;
    bool dofEnabled = false;
    bool phongEnabled = false;
    int opt;

     while ((opt = getopt(argc, argv, "s:a:pj")) != -1) {
        switch (opt) {
            case 's':
                numSamples = atoi(optarg); 
                break;
            case 'a':
                apertureSize = atof(optarg);
                dofEnabled = true;
                jitteringEnabled = true;
                break;
            case 'p':
                phongEnabled = true;
                break;
            case 'j':
                jitteringEnabled = true;
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


    int totalSamples = numSamples * numSamples; // Total number of rays per pixel
    std::cout << "-----------------------------------------\nUsing " << totalSamples << " samples per pixel" << std::endl;
    if (jitteringEnabled) {
        std::cout << "Jittering is enabled." << std::endl;
    } else {
        std::cout << "Jittering is disabled." << std::endl;
    }
    if (dofEnabled) {
        std::cout << "Depth of Field is enabled." << std::endl;
    } else {
        std::cout << "Depth of Field is disabled." << std::endl;
    }
    if (phongEnabled) {
        std::cout << "Phong is enabled." << std::endl;
    } else {
        std::cout << "Phong is disabled." << std::endl;
    }
    Scene scene;
    parser(nffFile, scene);
    printCameraInfo(scene.camera);
    std::cout << "Background Color: " << scene.background.transpose() << std::endl;
    for(const auto& light:scene.lights){
     std::cout << "Light: (" << light.position[0] << ", " << light.position[1] << ", " << light.position[2] << ")" << std::endl;
}
    for (const auto& material : scene.materials) {
    std::cout << "Material Properties:" << std::endl;
    std::cout << "Color: (" << material.color[0] << ", " << material.color[1] << ", " << material.color[2] << ")" << std::endl;
    std::cout << "Diffuse Coefficient (Kd): " << material.kd << std::endl;
    std::cout << "Specular Coefficient (Ks): " << material.ks << std::endl;
    std::cout << "Shininess (e): " << material.shine << std::endl; // or shine exponent
    std::cout << "Transmission Coefficient (Kt): " << material.transmittance << std::endl;
    std::cout << "Index of Refraction (IOR): " << material.ior << std::endl;
    std::cout << "------------------------------------" << std::endl; // Separator for readability
}
    std::cout << "Number of triangles parsed: " << scene.triangles.size() << std::endl;
    std::cout << "Number of spheres parsed: " << scene.spheres.size() << std::endl;
    std::cout << "Number of polygons parsed: " << scene.polygons.size() << std::endl;
    std::cout << "Number of polygon patches parsed: " << scene.polyPatches.size() << std::endl;
    std::cout << "Number or lights parsed: " << scene.lights.size() << std::endl;
    

    auto start = std::chrono::high_resolution_clock::now();

    render(scene, jitteringEnabled, numSamples, apertureSize, phongEnabled, outputFile); //THE RENDER

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nRender time: " << elapsed.count() << " seconds" << std::endl;
    std::cout<<"\nDone!"<<std::endl;


    return 0;

}
