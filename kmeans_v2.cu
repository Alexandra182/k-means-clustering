#include "stdio.h"
#include <chrono>
#include <fstream>
#include <iostream>

using namespace std;

#define kmean 100
#define H 1000
#define W 1000
#define iterations 200

__device__ float sqDistance(float *p1, float *p2) {
  return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) +
         (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

__global__ void kmeanKernel(float *pixels, float *cluster_pos,
                            float *sorteddata_pixels, float *cluster_meanpoint,
                            int *cluster_hitcount, int repetition) {

  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < H * W * 3) {
    if (i % 3 == 0) {

      float distance;
      float bestdistance;
      int bestcluster = 0;

      distance = INFINITY;
      bestdistance = distance;

      for (int m = 0; m < 3 * kmean; m = m + 3) {
        distance = sqDistance(cluster_pos + m, pixels + i);
        if (distance < bestdistance) {
          bestcluster = m / 3;
          bestdistance = distance;
        }
      }

      if (repetition == (iterations - 1)) {
        sorteddata_pixels[i] = cluster_pos[bestcluster * 3];
        sorteddata_pixels[i + 1] = cluster_pos[bestcluster * 3 + 1];
        sorteddata_pixels[i + 2] = cluster_pos[bestcluster * 3 + 2];
      }

      // __syncthreads();
      atomicAdd(&cluster_meanpoint[bestcluster * 3], pixels[i]);
      atomicAdd(&cluster_meanpoint[bestcluster * 3 + 1], pixels[i + 1]);
      atomicAdd(&cluster_meanpoint[bestcluster * 3 + 2], pixels[i + 2]);
      atomicAdd(&cluster_hitcount[bestcluster], 1);
    }
  }
}

__global__ void updateMeans(float *cluster_pos, float *pixels,
                            int *cluster_hitcount, float *meanpoint) {
  // The position is now the mean and the other parameters are restored
  int m = threadIdx.x + blockDim.x * blockIdx.x;

  if (m < 3 * kmean) {
    if (m % 3 == 0) {
      if (cluster_hitcount[m / 3] == 0) {
        // If no hit, then assign random value again
        cluster_pos[m] = pixels[m];
        cluster_pos[m + 1] = pixels[m + 1];
        cluster_pos[m + 2] = pixels[m + 2];
      }
      // Doing the mean of the meanpoint which contains the sum of all the hit
      // coordinates
      cluster_pos[m] = meanpoint[m] / (cluster_hitcount[m / 3]);
      cluster_pos[m + 1] = meanpoint[m + 1] / (cluster_hitcount[m / 3]);
      cluster_pos[m + 2] = meanpoint[m + 2] / (cluster_hitcount[m / 3]);

      // Restoring hit and meanpoint
      meanpoint[m] = 0;
      meanpoint[m + 1] = 0;
      meanpoint[m + 2] = 0;
      cluster_hitcount[m / 3] = 0;
    }
  }
}

int main() {
  ifstream inputValues("image.txt");

  float *host_pixels = 0;
  float *host_cluster_pos = 0;
  float *host_sorteddata_pixels = 0;
  float *host_meanpoint = 0;
  int *host_cluster_hitcount = 0;

  float *device_pixels = 0;
  float *device_cluster_pos = 0;
  float *device_sorteddata_pixels = 0;
  float *device_meanpoint = 0;
  int *device_cluster_hitcount = 0;

  host_pixels = (float *)malloc(3 * H * W * sizeof(float));
  host_cluster_pos = (float *)malloc(3 * kmean * sizeof(float));
  host_sorteddata_pixels = (float *)malloc(3 * H * W * sizeof(float));
  host_meanpoint = (float *)malloc(3 * kmean * sizeof(float));
  host_cluster_hitcount = (int *)malloc(kmean * sizeof(int));

  cudaMalloc((void **)&device_pixels, 3 * H * W * sizeof(float));
  cudaMalloc((void **)&device_cluster_pos, 3 * kmean * sizeof(float));
  cudaMalloc((void **)&device_sorteddata_pixels, 3 * H * W * sizeof(float));
  cudaMalloc((void **)&device_meanpoint, 3 * kmean * sizeof(float));
  cudaMalloc((void **)&device_cluster_hitcount, kmean * sizeof(int));

  for (int i = 0; i < 3 * H * W; ++i) {
    inputValues >> host_pixels[i];
  }

  inputValues.close();

  auto start = chrono::high_resolution_clock::now();

  // Asign random values to start the iteration
  for (int i = 0; i < 3 * kmean; i = i + 3) {
    host_cluster_pos[i] = host_pixels[i];
    host_cluster_pos[i + 1] = host_pixels[i + 1];
    host_cluster_pos[i + 2] = host_pixels[i + 2];
  }

  int block_size = 1024;
  int blocks_no = 3 * H * W / block_size;

  if (3 * H * W % block_size)
    ++blocks_no;

  int blocks_no2 = 3 * kmean / block_size;

  if (3 * kmean % block_size)
    ++blocks_no2;

  int repetition = 0;

  // printf("Blocks no, Blocks size: %d, %d\n", blocks_no, block_size);
  // printf("Blocks no, Blocks size: %d, %d\n", blocks_no2, block_size);

  cudaMemcpy(device_pixels, host_pixels, 3 * H * W * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_cluster_pos, host_cluster_pos, 3 * kmean * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_meanpoint, host_meanpoint, 3 * kmean * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_cluster_hitcount, host_cluster_hitcount,
             kmean * sizeof(int), cudaMemcpyHostToDevice);

  while (repetition < iterations) {
    repetition++;

    kmeanKernel<<<blocks_no, block_size>>>(
        device_pixels, device_cluster_pos, device_sorteddata_pixels,
        device_meanpoint, device_cluster_hitcount, repetition);

    cudaDeviceSynchronize();
    // printf("\n Error msg: %s \n", cudaGetErrorString(cudaGetLastError()));

    updateMeans<<<blocks_no2, block_size>>>(device_cluster_pos, device_pixels,
                                            device_cluster_hitcount,
                                            device_meanpoint);

    cudaDeviceSynchronize();
    // printf("\n Error msg: %s \n", cudaGetErrorString(cudaGetLastError()));
  }

  cudaMemcpy(host_sorteddata_pixels, device_sorteddata_pixels,
             3 * H * W * sizeof(float), cudaMemcpyDeviceToHost);

  // for (int i = 0; i < 3 * H * W; i = i + 3) {
  //   printf("%1.6f\t%1.6f\t%1.6f\n", host_sorteddata_pixels[i],
  //          host_sorteddata_pixels[i + 1], host_sorteddata_pixels[i + 2]);
  // }

  free(host_pixels);
  free(host_cluster_pos);
  free(host_sorteddata_pixels);
  free(host_meanpoint);
  free(host_cluster_hitcount);

  cudaFree(device_pixels);
  cudaFree(device_cluster_pos);
  cudaFree(device_sorteddata_pixels);
  cudaFree(device_meanpoint);
  cudaFree(device_cluster_hitcount);

  auto stop = chrono::high_resolution_clock::now();

  cout << "k means done in "
       << float(chrono::duration_cast<chrono::milliseconds>(stop - start)
                    .count() /
                1000.0)
       << " s" << endl;
}
