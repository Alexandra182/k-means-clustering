#include <Magick++.h>
#include <array>
#include <vector>

using namespace Magick;
using namespace std;

int main(int argc, char **argv) {
  int height = 1000;
  int width = 1000;

  std::vector<std::array<float, 3>> cloud1;
  cloud1.reserve(sizeof(std::array<float, 3>) * height * width);

  InitializeMagick(*argv);
  Image readimg("test.bmp");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ColorRGB rgb(readimg.pixelColor(i, j));
      cloud1[i * height + j][0] = rgb.red();
      cloud1[i * height + j][1] = rgb.green();
      cloud1[i * height + j][2] = rgb.blue();
      printf("%1.6f\t%1.6f\t%1.6f\n", cloud1[i * height + j][0],
             cloud1[i * height + j][1], cloud1[i * height + j][2]);
    }
  }
}