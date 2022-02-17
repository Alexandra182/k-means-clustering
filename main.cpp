#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

const int BYTES_PER_PIXEL = 3; /// red, green, & blue
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

void generateBitmapImage(unsigned char *image, int height, int width,
                         char *imageFileName);
unsigned char *createBitmapFileHeader(int height, int stride);
unsigned char *createBitmapInfoHeader(int height, int width);

using namespace std::chrono;
using namespace std;

int H = 1000, W = 1000;
int cloudlen = H * W;
int kmean = 15;
int iterations = 200;

ifstream inputValues("image.txt");

struct kcluster {
  std::array<float, 3> position;
  std::array<float, 3> meanpoint;
  int hitcount;
  float stdev;
};

void sqDistance(std::array<float, 3> p1, std::array<float, 3> p2,
                float &distance) {
  // distance = powf(p1[0] - p2[0], 2.0) + powf(p1[1] - p2[1], 2.0) +
  //            powf(p1[2] - p2[2], 2.0);
  distance = (p1[0] - p2[0]) * (p1[0] - p2[0]) +
             (p1[1] - p2[1]) * (p1[1] - p2[1]) +
             (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

void populateCloud(std::vector<std::array<float, 3>> &cloud1,
                   std::vector<std::array<float, 3>> &cloud2, int cloudlen) {
  for (int i = 0; i < cloudlen; i++) {
    for (int j = 0; j < 3; j++) {
      cloud2[i][j] = 20 * (rand() / (float)RAND_MAX) + 1;
      inputValues >> cloud1[i][j];
    }
  }
}

void printPoints(std::vector<std::array<float, 3>> cloud1,
                 std::vector<std::array<float, 3>> cloud2, int cloudlen) {
  for (int i = 0; i < cloudlen; i++) {
    std::cout << i << '\t' << cloud2[i][0] << '\t' << cloud2[i][1] << '\t'
              << cloud2[i][2] << '\n';
  }
}

void kmeansort(std::vector<std::array<float, 3>> &cloud1, int kmean,
               int cloudlen,
               std::vector<std::array<float, 3>> &sorteddata) {

  std::vector<kcluster> clustervect;
  clustervect.reserve(kmean * sizeof(kcluster));
  // Asign random values to start the iteration (hardcoded)
  for (int i = 0; i < kmean; i++) {
    clustervect[i].position[0] = cloud1[i][0];
    clustervect[i].position[1] = cloud1[i][1];
    clustervect[i].position[2] = cloud1[i][2];
  }
  // While the k-mean is not sufficcient we keep iterating
  bool conformitycondition = true;
  int repetition = 0;
  while (repetition < iterations) {
    repetition++;
    // In every point check distance to closest cluster
    for (int j = 0; j < cloudlen; j++) {
      float distance;
      float bestdistance;
      distance = INFINITY;
      bestdistance = distance;
      int bestcluster = 0;
      for (int m = 0; m < kmean; m++) {
        sqDistance(clustervect[m].position, cloud1[j], distance);
        if (distance < bestdistance) {
          bestcluster = m;
          bestdistance = distance;
        }
      }
      if (repetition == (iterations - 1)) {
        // // If this is the last iteration, print it to the user
        // printf("%1.6f\t%1.6f\t%1.6f\n", clustervect[bestcluster].position[0],
        //        clustervect[bestcluster].position[1],
        //        clustervect[bestcluster].position[2]);
        sorteddata[j][0] = clustervect[bestcluster].position[0];
        sorteddata[j][1] = clustervect[bestcluster].position[1];
        sorteddata[j][2] = clustervect[bestcluster].position[2];
      }
      // Asign the point to the closest cluster
      clustervect[bestcluster].meanpoint[0] += cloud1[j][0];
      clustervect[bestcluster].meanpoint[1] += cloud1[j][1];
      clustervect[bestcluster].meanpoint[2] += cloud1[j][2];
      clustervect[bestcluster].hitcount++;
    }

    // The position is now the mean and the other parameters are restored
    for (int m = 0; m < kmean; m++) {
      if (clustervect[m].hitcount == 0) {
        // If no hit, then assign random value again
        clustervect[m].position[0] = cloud1[m][0];
        clustervect[m].position[1] = cloud1[m][1];
        clustervect[m].position[2] = cloud1[m][2];
      }
      // Doing the mean of the meanpoint which contains the sum of all the hit
      // coordinates
      clustervect[m].position[0] =
          clustervect[m].meanpoint[0] / (clustervect[m].hitcount);
      clustervect[m].position[1] =
          clustervect[m].meanpoint[1] / (clustervect[m].hitcount);
      clustervect[m].position[2] =
          clustervect[m].meanpoint[2] / (clustervect[m].hitcount);

      // Restoring hit and meanpoint
      clustervect[m].meanpoint[0] = 0;
      clustervect[m].meanpoint[1] = 0;
      clustervect[m].meanpoint[2] = 0;
      clustervect[m].hitcount = 0;
    }
  }
}

int main(void) {
  // Declaring the dataset
  std::vector<std::array<float, 3>> cloud1;
  std::vector<std::array<float, 3>> cloud2;
  std::vector<std::array<float, 3>> sorteddata;

  cloud1.reserve(sizeof(std::array<float, 3>) * cloudlen);
  cloud2.reserve(sizeof(std::array<float, 3>) * cloudlen);
  sorteddata.reserve(sizeof(std::array<float, 3>) * cloudlen);

  // populating the datasets
  populateCloud(cloud1, cloud2, cloudlen);

  auto start = std::chrono::high_resolution_clock::now();
  kmeansort(cloud1, kmean, cloudlen, sorteddata);
  auto stop = std::chrono::high_resolution_clock::now();

  cout << "k means done in "
       << float(chrono::duration_cast<chrono::milliseconds>(stop - start)
                    .count() /
                1000.0)
       << " s" << endl;

  inputValues.close();

  std::vector<std::array<float, 3>> image_pixels;

  unsigned char image[H][W][BYTES_PER_PIXEL];
  char *imageFileName = (char *)"cpu.bmp";

  int index = 0;
  for (int i = 0; i < H; i++) {
    for (int j = W - 1; j >= 0; j--) {
      cloud1[i * H + j][0] = sorteddata[index][0];
      cloud1[i * H + j][1] = sorteddata[index][1];
      cloud1[i * H + j][2] = sorteddata[index][2];
      index += 1;
    }
  }

  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      image[j][i][2] = (unsigned char)(cloud1[i * H + j][0] * 255); /// red
      image[j][i][1] =
          (unsigned char)(cloud1[i * H + j][1] * 255); /// green
      image[j][i][0] =
          (unsigned char)(cloud1[i * H + j][2] * 255); /// blue
    }
  }

  generateBitmapImage((unsigned char *)image, H, W, imageFileName);
  printf("Image generated!!");
}

void generateBitmapImage(unsigned char *image, int height, int width,
                         char *imageFileName) {
  int widthInBytes = width * BYTES_PER_PIXEL;

  unsigned char padding[3] = {0, 0, 0};
  int paddingSize = (4 - (widthInBytes) % 4) % 4;

  int stride = (widthInBytes) + paddingSize;

  FILE *imageFile = fopen(imageFileName, "wb");

  unsigned char *fileHeader = createBitmapFileHeader(height, stride);
  fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

  unsigned char *infoHeader = createBitmapInfoHeader(height, width);
  fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

  int i;
  for (i = 0; i < height; i++) {
    fwrite(image + (i * widthInBytes), BYTES_PER_PIXEL, width, imageFile);
    fwrite(padding, 1, paddingSize, imageFile);
  }

  fclose(imageFile);
}

unsigned char *createBitmapFileHeader(int height, int stride) {
  int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

  static unsigned char fileHeader[] = {
      0, 0,       /// signature
      0, 0, 0, 0, /// image file size in bytes
      0, 0, 0, 0, /// reserved
      0, 0, 0, 0, /// start of pixel array
  };

  fileHeader[0] = (unsigned char)('B');
  fileHeader[1] = (unsigned char)('M');
  fileHeader[2] = (unsigned char)(fileSize);
  fileHeader[3] = (unsigned char)(fileSize >> 8);
  fileHeader[4] = (unsigned char)(fileSize >> 16);
  fileHeader[5] = (unsigned char)(fileSize >> 24);
  fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

  return fileHeader;
}

unsigned char *createBitmapInfoHeader(int height, int width) {
  static unsigned char infoHeader[] = {
      0, 0, 0, 0, /// header size
      0, 0, 0, 0, /// image width
      0, 0, 0, 0, /// image height
      0, 0,       /// number of color planes
      0, 0,       /// bits per pixel
      0, 0, 0, 0, /// compression
      0, 0, 0, 0, /// image size
      0, 0, 0, 0, /// horizontal resolution
      0, 0, 0, 0, /// vertical resolution
      0, 0, 0, 0, /// colors in color table
      0, 0, 0, 0, /// important color count
  };

  infoHeader[0] = (unsigned char)(INFO_HEADER_SIZE);
  infoHeader[4] = (unsigned char)(width);
  infoHeader[5] = (unsigned char)(width >> 8);
  infoHeader[6] = (unsigned char)(width >> 16);
  infoHeader[7] = (unsigned char)(width >> 24);
  infoHeader[8] = (unsigned char)(height);
  infoHeader[9] = (unsigned char)(height >> 8);
  infoHeader[10] = (unsigned char)(height >> 16);
  infoHeader[11] = (unsigned char)(height >> 24);
  infoHeader[12] = (unsigned char)(1);
  infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL * 8);

  return infoHeader;
}