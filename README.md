# k-means-clustering
Parallelised version of the k-means clustering algorithm for colour reduction using CUDA.

Based on [lasdasdas](https://github.com/lasdasdas)'s serial implementation: https://github.com/lasdasdas/k-means-clustering

<br/>
<table>
  <tr>
    <td align="center"><img src="img/original.jpg" alt="1" width = 300px height = 300px ></td>
    <td align="center"><img src="img/reduced.jpg" alt="2" width = 300px height = 300px></td>
   </tr> 
   <tr>
    <td align="center">Original</td>
    <td align="center">Reduced number of colours</td>
  </td>
  </tr>
</table>

## Prerequisites
- CUDA 11.4

## Building the project
Generate RGB pixel values from the image:
```
g++ readPixels.cpp `Magick++-config --cxxflags --cppflags --ldflags --libs` -o readPixels
./readPixels > image.txt
```

Apply the k-means clustering algorithm on the generated pixels (reduce the number of colours in the image to a number specified by the *kmeans* parameter):

- Serial implementation
```
g++ main.cpp -o kmeanCPU -O3 -std=gnu++11  
```

- Parallel implementation (CUDA)
```
nvcc main.cu -o kmeanCUDA -std=c++11
```



