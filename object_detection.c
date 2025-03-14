#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

typedef struct {
    float x;
    float y;
} Point;

typedef struct {
    Point *points;
    int numPoints;
} PointCloud;

typedef struct {
    float tx;  // translation in x
    float ty;  // translation in y
} Transformation;

// -----------------------------------------------------------------------------
// Image Loading (Grayscale) via stb_image
// -----------------------------------------------------------------------------

unsigned char* loadImage(const char *filename, int *width, int *height) {
    int channels_in_file;
    unsigned char *data = stbi_load(filename, width, height, &channels_in_file, 1);
    if (!data) {
        fprintf(stderr, "Error: Could not load image '%s'\n", filename);
        return NULL;
    }
    return data;
}

// -----------------------------------------------------------------------------
// Sobel Edge Detection
// -----------------------------------------------------------------------------

void sobelEdgeDetector(const unsigned char *img, int width, int height,
                       unsigned char *edgeOutput, float threshold)
{
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };
    
    memset(edgeOutput, 0, width * height);
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sumX = 0.0f;
            float sumY = 0.0f;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    float pixelVal = (float)img[py * width + px];
                    
                    sumX += pixelVal * Gx[ky + 1][kx + 1];
                    sumY += pixelVal * Gy[ky + 1][kx + 1];
                }
            }
            float magnitude = sqrtf(sumX * sumX + sumY * sumY);
            if (magnitude > threshold) {
                edgeOutput[y * width + x] = 255;
            } else {
                edgeOutput[y * width + x] = 0;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Convert Edge Map to Point Cloud
// -----------------------------------------------------------------------------

typedef struct {
    Point *points;
    int numPoints;
} EdgePoints;

PointCloud createPointCloudFromEdges(const unsigned char *edgeMap, int width, int height) {
    // Count how many edges
    int count = 0;
    for (int i = 0; i < width * height; i++) {
        if (edgeMap[i] == 255) count++;
    }
    PointCloud pc;
    pc.numPoints = count;
    pc.points = (Point *)malloc(count * sizeof(Point));
    
    int idx = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (edgeMap[y * width + x] == 255) {
                pc.points[idx].x = (float)x;
                pc.points[idx].y = (float)y;
                idx++;
            }
        }
    }
    return pc;
}

// -----------------------------------------------------------------------------
// Compute Distance Transform (2D) for the scene edge map
// -----------------------------------------------------------------------------

/*
    We compute the distance transform using a common two-pass algorithm:
    1) Forward pass (top-left to bottom-right)
    2) Backward pass (bottom-right to top-left)
    
    Each pixel gets the distance to the nearest edge pixel (0 if it's an edge).
    We'll store distances in a float array distMap[width*height].
*/

static inline float minf(float a, float b) { return (a < b) ? a : b; }

void computeDistanceTransform(const unsigned char *edgeMap, int width, int height, float *distMap) {
    // Initialize distMap
    // If it's an edge pixel => distance = 0
    // Otherwise => large number (e.g., width+height)
    float maxDist = (float)(width + height);
    for (int i = 0; i < width * height; i++) {
        distMap[i] = (edgeMap[i] == 255) ? 0.0f : maxDist;
    }
    
    // 1) Forward pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float current = distMap[y * width + x];
            // Check left neighbor
            if (x > 0) {
                float left = distMap[y * width + (x-1)] + 1.0f;
                if (left < current) {
                    current = left;
                }
            }
            // Check top neighbor
            if (y > 0) {
                float top = distMap[(y-1) * width + x] + 1.0f;
                if (top < current) {
                    current = top;
                }
            }
            distMap[y * width + x] = current;
        }
    }
    
    // 2) Backward pass
    for (int y = height - 1; y >= 0; y--) {
        for (int x = width - 1; x >= 0; x--) {
            float current = distMap[y * width + x];
            // Check right neighbor
            if (x < width - 1) {
                float right = distMap[y * width + (x+1)] + 1.0f;
                if (right < current) {
                    current = right;
                }
            }
            // Check bottom neighbor
            if (y < height - 1) {
                float bottom = distMap[(y+1) * width + x] + 1.0f;
                if (bottom < current) {
                    current = bottom;
                }
            }
            distMap[y * width + x] = current;
        }
    }
}

// -----------------------------------------------------------------------------
// Directed Hausdorff Distance Using the Scene's Distance Transform
// -----------------------------------------------------------------------------

/*
    For each point in the transformed model, we look up its distance
    in the scene's distMap. Then we take the maximum of these distances.
*/

float directedHausdorffUsingDT(PointCloud model, float *distMap, int sceneWidth, int sceneHeight) {
    float maxDist = 0.0f;
    for (int i = 0; i < model.numPoints; i++) {
        int ix = (int)(model.points[i].x + 0.5f);
        int iy = (int)(model.points[i].y + 0.5f);
        
        // Check if (ix, iy) is inside the scene
        if (ix < 0 || ix >= sceneWidth || iy < 0 || iy >= sceneHeight) {
            // If outside, define a large penalty, e.g., distance = some big number
            float penalty = 50.0f; // or sceneWidth+sceneHeight
            if (penalty > maxDist) {
                maxDist = penalty;
            }
        } else {
            float d = distMap[iy * sceneWidth + ix];
            if (d > maxDist) {
                maxDist = d;
            }
        }
    }
    return maxDist;
}

// -----------------------------------------------------------------------------
// Simple Translation Search Using Distance Transform
// -----------------------------------------------------------------------------

PointCloud translatePointCloud(PointCloud pc, float tx, float ty) {
    PointCloud result;
    result.numPoints = pc.numPoints;
    result.points = (Point *)malloc(result.numPoints * sizeof(Point));
    for (int i = 0; i < pc.numPoints; i++) {
        result.points[i].x = pc.points[i].x + tx;
        result.points[i].y = pc.points[i].y + ty;
    }
    return result;
}

void findBestTranslationDT(PointCloud objectPC, float *sceneDistMap,
                           int sceneWidth, int sceneHeight,
                           float minTx, float maxTx, float stepTx,
                           float minTy, float maxTy, float stepTy)
{
    float bestScore = FLT_MAX;
    float bestTx = 0.0f;
    float bestTy = 0.0f;
    
    // Exhaustive search over translations
    for (float tx = minTx; tx <= maxTx; tx += stepTx) {
        for (float ty = minTy; ty <= maxTy; ty += stepTy) {
            // Translate object
            PointCloud transObj = translatePointCloud(objectPC, tx, ty);
            // Compute directed Hausdorff distance from object->scene using dist transform
            float score = directedHausdorffUsingDT(transObj, sceneDistMap, sceneWidth, sceneHeight);
            free(transObj.points);
            
            if (score < bestScore) {
                bestScore = score;
                bestTx = tx;
                bestTy = ty;
            }
        }
    }
    printf("Best Score = %f at translation (%.2f, %.2f)\n", bestScore, bestTx, bestTy);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s <scene_image> <object_image>\n", argv[0]);
        return 1;
    }
    
    const char *sceneFile = argv[1];
    const char *objectFile = argv[2];
    
    // 1) Load Scene
    int sw, sh;
    unsigned char *sceneData = loadImage(sceneFile, &sw, &sh);
    if (!sceneData) {
        return 1;
    }
    // 2) Load Object
    int ow, oh;
    unsigned char *objectData = loadImage(objectFile, &ow, &oh);
    if (!objectData) {
        free(sceneData);
        return 1;
    }
    
    // 3) Sobel Edge Detection
    unsigned char *sceneEdges = (unsigned char *)calloc(sw*sh, sizeof(unsigned char));
    unsigned char *objectEdges = (unsigned char *)calloc(ow*oh, sizeof(unsigned char));
    
    float threshold = 100.0f; // adjust as needed
    sobelEdgeDetector(sceneData, sw, sh, sceneEdges, threshold);
    sobelEdgeDetector(objectData, ow, oh, objectEdges, threshold);
    
    // 4) Create Point Clouds
    PointCloud scenePC = createPointCloudFromEdges(sceneEdges, sw, sh);
    PointCloud objectPC = createPointCloudFromEdges(objectEdges, ow, oh);
    printf("Scene edges: %d points\n", scenePC.numPoints);
    printf("Object edges: %d points\n", objectPC.numPoints);
    
    // 5) Compute Distance Transform of the SCENE edges
    float *sceneDistMap = (float *)malloc(sw * sh * sizeof(float));
    computeDistanceTransform(sceneEdges, sw, sh, sceneDistMap);
    
    // 6) Simple Translation Search using the Scene's Distance Transform
    float minTx = -50, maxTx = 50, stepTx = 2;
    float minTy = -50, maxTy = 50, stepTy = 2;
    findBestTranslationDT(objectPC, sceneDistMap, sw, sh,
                          minTx, maxTx, stepTx,
                          minTy, maxTy, stepTy);
    
    // Cleanup
    free(sceneData);
    free(objectData);
    free(sceneEdges);
    free(objectEdges);
    free(scenePC.points);
    free(objectPC.points);
    free(sceneDistMap);
    
    return 0;
}
