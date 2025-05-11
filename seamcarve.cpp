
#define cimg_display 0
#include "CImg.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <getopt.h>
using namespace cimg_library;

//prototypes here, not really


/** FUNCTIONS BELOW **/

//compute energy based on the gradient magnitude at each pixel
std::vector<std::vector<double>> computeEnergy(const CImg<double>& img) {
    int width = img.width();
    int height = img.height();
    std::vector<std::vector<double>> energy(height, std::vector<double>(width, 0.0));
    double max_energy = 0.0;

    // Calculate energy for each pixel based on gradients in x and y directions
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double dx = 0.0, dy = 0.0;
            //bounds
            for (int c = 0; c < img.spectrum(); ++c) { // Loop through color channels
                double left = img(x > 0 ? x - 1 : x, y, 0, c);
                double right = img(x < width - 1 ? x + 1 : x, y, 0, c);
                double up = img(x, y > 0 ? y - 1 : y, 0, c);
                double down = img(x, y < height - 1 ? y + 1 : y, 0, c);

                dx += std::pow(right - left, 2);
                dy += std::pow(down - up, 2);
            }
            energy[y][x] = std::sqrt(dx + dy); // Gradient magnitude
            max_energy = std::max(max_energy, energy[y][x]); // Track max energy for normalization
        }
    }

    // Normalize energy values to the range [0, 1]
    if (max_energy > 0.0) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                energy[y][x] /= max_energy;
            }
        }
    }

    return energy;
}


std::vector<std::vector<double>> computeVisualSaliencyEnergy(const CImg<double>& img) {
    int width = img.width();
    int height = img.height();
    std::vector<std::vector<double>> energy(height, std::vector<double>(width, 0.0));

    // Create a blurred version of the image
    CImg<double> blurredImage = img.get_blur(5.0); // Gaussian blur with sigma = 5.0

    double max_energy = 0.0;
    
    // Compute saliency as the difference between original and blurred images
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double saliency = 0.0;
            for (int c = 0; c < img.spectrum(); ++c) {
                double original_pixel = img(x, y, 0, c);
                double blurred_pixel = blurredImage(x, y, 0, c);
                saliency += std::pow(original_pixel - blurred_pixel, 2);
            }
            energy[y][x] = std::sqrt(saliency);
            max_energy = std::max(max_energy, energy[y][x]); // Track max energy for normalization
        }
    }

    // Normalize energy values to [0, 1]
    if (max_energy > 0.0) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                energy[y][x] /= max_energy;
            }
        }
    }

    return energy;
}

std::vector<std::vector<double>> computeScharrGradientEnergy(const CImg<double>& img) {
    int width = img.width();
    int height = img.height();
    std::vector<std::vector<double>> energy(height, std::vector<double>(width, 0.0));
    double max_energy = 0.0;

    // Scharr filters for x and y gradients
    const int Gx[3][3] = { { 47,  0,  -47 }, 
                           { 162,  0, -162 }, 
                           { 47,  0,  -47 } };

    const int Gy[3][3] = { {  47,  162,  47 }, 
                           {  0,   0,  0 }, 
                           { -47, -162, -47 } };

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            double gx = 0.0, gy = 0.0;
            
            // Apply Scharr filter for each color channel
            for (int c = 0; c < img.spectrum(); ++c) {
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        double pixel = img(x + j, y + i, 0, c);
                        gx += pixel * Gx[i + 1][j + 1];
                        gy += pixel * Gy[i + 1][j + 1];
                    }
                }
            }

            energy[y][x] = std::sqrt(gx * gx + gy * gy); // Gradient magnitude
            max_energy = std::max(max_energy, energy[y][x]); // Track max energy for normalization
        }
    }

    // Normalize energy values to [0, 1]
    if (max_energy > 0.0) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                energy[y][x] /= max_energy;
            }
        }
    }

    return energy;
}

std::vector<std::vector<double>> computeCombinedEnergy(const CImg<double>& img) {
    auto gradientEnergy = computeEnergy(img);
    auto saliencyEnergy = computeVisualSaliencyEnergy(img);
    auto scharrEnergy = computeScharrGradientEnergy(img);

    int height = img.height();
    int width = img.width();
    std::vector<std::vector<double>> combinedEnergy(height, std::vector<double>(width, 0.0));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            combinedEnergy[y][x] = std::max({gradientEnergy[y][x], saliencyEnergy[y][x], scharrEnergy[y][x]});
        }
    }

    return combinedEnergy;
}

//Only recompute the energy of pixels affected by the seam removal
void updateEnergy(const CImg<double>& img, std::vector<std::vector<double>>& energy, const std::vector<int>& seam, bool isVertical) {
    int width = img.width();
    int height = img.height();

    if (isVertical) {
        // Update energy for pixels in a vertical seam and their neighbors
        for (int y = 0; y < height; ++y) {
            int xStart = std::max(0, seam[y] - 1);
            int xEnd = std::min(width - 1, seam[y] + 1);
            for (int x = xStart; x <= xEnd; ++x) {
                double dx = 0.0, dy = 0.0;
                for (int c = 0; c < img.spectrum(); ++c) {
                    double left = img(x > 0 ? x - 1 : x, y, 0, c);
                    double right = img(x < width - 1 ? x + 1 : x, y, 0, c);
                    double up = img(x, y > 0 ? y - 1 : y, 0, c);
                    double down = img(x, y < height - 1 ? y + 1 : y, 0, c);

                    dx += std::pow(right - left, 2);
                    dy += std::pow(down - up, 2);
                }
                energy[y][x] = std::sqrt(dx + dy);
            }
        }
    } else {
        // Update energy for pixels in a horizontal seam and their neighbors
        for (int x = 0; x < width; ++x) {
            int yStart = std::max(0, seam[x] - 1);
            int yEnd = std::min(height - 1, seam[x] + 1);
            for (int y = yStart; y <= yEnd; ++y) {
                double dx = 0.0, dy = 0.0;
                for (int c = 0; c < img.spectrum(); ++c) {
                    double left = img(x > 0 ? x - 1 : x, y, 0, c);
                    double right = img(x < width - 1 ? x + 1 : x, y, 0, c);
                    double up = img(x, y > 0 ? y - 1 : y, 0, c);
                    double down = img(x, y < height - 1 ? y + 1 : y, 0, c);

                    dx += std::pow(right - left, 2);
                    dy += std::pow(down - up, 2);
                }
                energy[y][x] = std::sqrt(dx + dy);
            }
        }
    }
}

// Find the optimal vertical seam to remove based on the energy map
std::vector<int> findVerticalSeam(const std::vector<std::vector<double>>& energy) {
    int height = energy.size();
    int width = energy[0].size();
    std::vector<std::vector<double>> dp(height, std::vector<double>(width, 0.0)); // DP table
    std::vector<std::vector<int>> backtrack(height, std::vector<int>(width, 0)); // Backtracking array

    // Initialize the first row of DP table with energy values
    for (int x = 0; x < width; ++x) dp[0][x] = energy[0][x];

    // Fill the DP table
    for (int y = 1; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dp[y][x] = energy[y][x];
            backtrack[y][x] = x;
            double minEnergy = dp[y - 1][x];
            if (x > 0 && dp[y - 1][x - 1] < minEnergy) {
                minEnergy = dp[y - 1][x - 1];
                backtrack[y][x] = x - 1;
            }
            if (x < width - 1 && dp[y - 1][x + 1] < minEnergy) {
                minEnergy = dp[y - 1][x + 1];
                backtrack[y][x] = x + 1;
            }
            dp[y][x] += minEnergy;
        }
    }

    // Trace back the minimum energy path (vertical seam)
    std::vector<int> seam(height);
    int minIndex = std::min_element(dp[height - 1].begin(), dp[height - 1].end()) - dp[height - 1].begin();
    for (int y = height - 1; y >= 0; --y) {
        seam[y] = minIndex;
        minIndex = backtrack[y][minIndex];
    }
    return seam;
}


// Remove Vertical Seam
CImg<double> removeVerticalSeam(CImg<double>& img, std::vector<std::vector<double>>& energy, const std::vector<int>& seam) {
    int width = img.width();
    int height = img.height();
    CImg<double> result(width - 1, height, img.depth(), img.spectrum(), 0);

    for (int y = 0; y < height; ++y) {
        int targetX = 0;
        for (int x = 0; x < width; ++x) {
            if (x != seam[y]) {
                for (int c = 0; c < img.spectrum(); ++c) {
                    result(targetX, y, 0, c) = img(x, y, 0, c);
                }
                ++targetX;
            }
        }
    }

    // Update the energy map locally around the removed seam
    updateEnergy(result, energy, seam, true);

    return result;
}

// Find Horizontal Seam
std::vector<int> findHorizontalSeam(const std::vector<std::vector<double>>& energy) {
    int height = energy.size();
    int width = energy[0].size();
    std::vector<std::vector<double>> dp(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<int>> backtrack(height, std::vector<int>(width, 0));

    for (int y = 0; y < height; ++y) dp[y][0] = energy[y][0];
    for (int x = 1; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            dp[y][x] = energy[y][x];
            backtrack[y][x] = y;
            double minEnergy = dp[y][x - 1];
            if (y > 0 && dp[y - 1][x - 1] < minEnergy) {
                minEnergy = dp[y - 1][x - 1];
                backtrack[y][x] = y - 1;
            }
            if (y < height - 1 && dp[y + 1][x - 1] < minEnergy) {
                minEnergy = dp[y + 1][x - 1];
                backtrack[y][x] = y + 1;
            }
            dp[y][x] += minEnergy;
        }
    }

    std::vector<int> seam(width);
    int minIndex = std::min_element(dp.begin(), dp.end(), [&](const auto& a, const auto& b) { return a.back() < b.back(); }) - dp.begin();
    for (int x = width - 1; x >= 0; --x) {
        seam[x] = minIndex;
        minIndex = backtrack[minIndex][x];
    }
    return seam;
}

// Remove Horizontal Seam
CImg<double> removeHorizontalSeam(CImg<double>& img, std::vector<std::vector<double>>& energy, const std::vector<int>& seam) {
    int width = img.width();
    int height = img.height();
    CImg<double> result(width, height - 1, img.depth(), img.spectrum(), 0);

    for (int x = 0; x < width; ++x) {
        int targetY = 0;
        for (int y = 0; y < height; ++y) {
            if (y != seam[x]) {
                for (int c = 0; c < img.spectrum(); ++c) {
                    result(x, targetY, 0, c) = img(x, y, 0, c);
                }
                ++targetY;
            }
        }
    }

    // Update the energy map locally around the removed seam
    updateEnergy(result, energy, seam, false);

    return result;
}


//outputs an energy map
void saveEnergyImage(const std::vector<std::vector<double>>& energy, const char* outputFile) {
    int height = energy.size();
    int width = energy[0].size();

    // Find the maximum energy value for normalization
    double max_energy = 0.0;
    for (const auto& row : energy) {
        for (double e : row) {
            max_energy = std::max(max_energy, e);
        }
    }

    // Create a grayscale image for the energy map
    CImg<unsigned char> energyImage(width, height, 1, 1, 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double normalized = std::pow(energy[y][x] / max_energy, 1.0 / 3.0);
            unsigned char intensity = static_cast<unsigned char>(normalized * 255);
            energyImage(x, y) = intensity;
        }
    }

    // Save the energy map image
    energyImage.save(outputFile);
    std::cout << "Energy map saved to: " << outputFile << "\n";
}

void displayProgress(int processedSeams, int totalSeams) {
    int progress = static_cast<int>(100.0 * processedSeams / totalSeams);
    std::cout << "\rProgress: " << progress << "%";
    std::cout.flush();
}

void validateDimensions(int targetWidth, int targetHeight, int imageWidth, int imageHeight) {
    if (targetWidth < 0 || targetHeight < 0) {
        throw std::invalid_argument("Target dimensions must be non-negative.");
    }
    if (targetWidth > imageWidth || targetHeight > imageHeight) {
        throw std::invalid_argument("Target dimensions must be less than or equal to the original dimensions.");
    }
}

CImg<double> processImage(CImg<double>& input, int targetWidth, int targetHeight) {
    validateDimensions(targetWidth, targetHeight, input.width(), input.height());
    int originalWidth = input.width();
    int originalHeight = input.height();

    int totalVerticalSeams = originalWidth - targetWidth;
    int totalHorizontalSeams = originalHeight - targetHeight;
    int totalSeams = totalVerticalSeams + totalHorizontalSeams;
    int processedSeams = 0;

    while (input.width() > targetWidth || input.height() > targetHeight) {
        auto energy = computeCombinedEnergy(input);

        //Remove vertical or horizontal seam based on remaining dimensions
        if (input.width() > targetWidth) {
            auto seam = findVerticalSeam(energy);
            input = removeVerticalSeam(input, energy, seam);
            processedSeams++;
        }
        if (input.height() > targetHeight) {
            auto seam = findHorizontalSeam(energy);
            input = removeHorizontalSeam(input, energy, seam);
            processedSeams++;
        }

        displayProgress(processedSeams, totalSeams);
    }

    std::cout << "\nProcessing completed.\n";
    return input;
}



int main(int argc, char* argv[]) {
    const char* inputFile = nullptr;
    const char* outputFile = nullptr;
    int targetWidth = -1;
    int targetHeight = -1;
    bool saveEnergy = false;

    // Parse arguments
    int opt;
    while ((opt = getopt(argc, argv, "e")) != -1) {
        switch (opt) {
            case 'e':
                saveEnergy = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-e] input_image[.jpg/png] output_image target_width target_height\n";
                return 1;
        }
    }

    // Ensure correct number of positional arguments
    if (argc - optind < 4) {
        std::cerr << "Usage: " << argv[0] << " [-e] input_image output_image target_width target_height\n";
        return 1;
    }

    inputFile = argv[optind];
    outputFile = argv[optind + 1];
    targetWidth = std::stoi(argv[optind + 2]);
    targetHeight = std::stoi(argv[optind + 3]);

    try {
        // Load input image
        CImg<double> inputImage(inputFile);

        // Compute and save energy map
        if (saveEnergy) {
            auto energy = computeCombinedEnergy(inputImage);
            saveEnergyImage(energy, "energy_map.png");
        }

        // Resize image
        CImg<double> outputImage = processImage(inputImage, targetWidth, targetHeight);

        // Save the output image
        outputImage.save(outputFile);
        std::cout << "Output saved to: " << outputFile << "\n";
    } catch (const cimg_library::CImgIOException& e) {
        std::cerr << "Error loading image file: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
