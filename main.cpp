#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std::chrono;
using namespace std;

int H = 300, W = 300;
int cloudlen = H * W;
int kmean = 10;
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
  distance = powf(p1[0] - p2[0], 2.0) + powf(p1[1] - p2[1], 2.0) +
             powf(p1[2] - p2[2], 2.0);
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

//
void printPoints(std::vector<std::array<float, 3>> cloud1,
                 std::vector<std::array<float, 3>> cloud2, int cloudlen) {
  for (int i = 0; i < cloudlen; i++) {
    std::cout << i << '\t' << cloud2[i][0] << '\t' << cloud2[i][1] << '\t'
              << cloud2[i][2] << '\n';
  }
}

void kmeansort(std::vector<std::array<float, 3>> &cloud1, int kmean,
               int cloudlen,
               std::vector<std::vector<std::array<float, 3>>> &sorteddata) {

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
      sqDistance(clustervect[0].position, cloud1[j], distance);
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
        // If this is the last iteration, print it to the user

        printf("%1.6f\t%1.6f\t%1.6f\n", clustervect[bestcluster].position[0],
               clustervect[bestcluster].position[1],
               clustervect[bestcluster].position[2]);
        sorteddata[bestcluster].push_back(cloud1[j]);
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
  cloud1.reserve(sizeof(std::array<float, 3>) * cloudlen);
  cloud2.reserve(sizeof(std::array<float, 3>) * cloudlen);
  std::vector<std::vector<std::array<float, 3>>> sorteddata(kmean);
  // populating the datasets
  populateCloud(cloud1, cloud2, cloudlen);

  auto start2 = std::chrono::high_resolution_clock::now();
  kmeansort(cloud1, kmean, cloudlen, sorteddata);
  auto stop2 = std::chrono::high_resolution_clock::now();
  // std::cout << "k means done in " <<
  // duration_cast<std::chrono::seconds>(stop2 - start2).count() << " s" <<
  // std::endl;

  inputValues.close();
}
