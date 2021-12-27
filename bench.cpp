/*
MIT License

Copyright (c) 2021 Alex Emirov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "flatbush.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using randomDouble = std::uniform_real_distribution<double>;

void addRandomBox(std::vector<double>& iBoxes, size_t iBoxSize)
{
  std::random_device wDevice;
  std::mt19937 wEngine(wDevice());
  double wMinX = randomDouble(0.0, 100 - iBoxSize)(wEngine);
  double wMinY = randomDouble(0.0, 100 - iBoxSize)(wEngine);
  double wMaxX = wMinX + randomDouble(0.0, iBoxSize)(wEngine);
  double wMaxY = wMinY + randomDouble(0.0, iBoxSize)(wEngine);
  iBoxes.push_back(wMinX);
  iBoxes.push_back(wMinY);
  iBoxes.push_back(wMaxX);
  iBoxes.push_back(wMaxY);
}

void benchSearch(const flatbush::Flatbush<double>& iIndex, const std::vector<double>& iBoxes, size_t iNumTests, double iPercentage)
{
  auto wStartTime = std::chrono::high_resolution_clock::now();

  for (size_t wIdx = 0; wIdx < iBoxes.size(); wIdx += 4) {
    iIndex.search(iBoxes[wIdx], iBoxes[wIdx + 1], iBoxes[wIdx + 2], iBoxes[wIdx + 3]);
  }

  auto wEndTime = std::chrono::high_resolution_clock::now();
  std::cout << iNumTests << " searches " << iPercentage << "%: " << std::chrono::duration_cast<std::chrono::milliseconds>((wEndTime - wStartTime)).count() << "ms" << std::endl;
}

void benchNeighbors(const flatbush::Flatbush<double>& iIndex, const std::vector<double>& iCoords, size_t iNumTests, size_t iNeighbors)
{
  auto wStartTime = std::chrono::high_resolution_clock::now();

  for (size_t wIdx = 0; wIdx < iNumTests; ++wIdx) {
    iIndex.neighbors(iCoords[4 * wIdx], iCoords[4 * wIdx + 1], iNeighbors);
  }

  auto wEndTime = std::chrono::high_resolution_clock::now();
  std::cout << iNumTests << " searches of " << iNeighbors << " neighbors: " << std::chrono::duration_cast<std::chrono::milliseconds>((wEndTime - wStartTime)).count() << "ms" << std::endl;
}

int main(int argc, char** argv)
{
  size_t wNumItems = 1000000;
  size_t wNumTests = 1000;
  size_t wNodeSize = 16;

  std::vector<double> wCoords;
  for (size_t wCount = 0; wCount < wNumItems; ++wCount)
  {
    addRandomBox(wCoords, 1);
  }

  std::vector<double> wBoxes100;
  std::vector<double> wBoxes10;
  std::vector<double> wBoxes1;
  for (size_t wCount = 0; wCount < wNumTests; ++wCount)
  {
    addRandomBox(wBoxes100, 100 * std::sqrt(0.1));
    addRandomBox(wBoxes10, 10);
    addRandomBox(wBoxes1, 1);
  }

  auto wStartTime = std::chrono::high_resolution_clock::now();

  auto wIndex = flatbush::Flatbush<double>::create(wNumItems, wNodeSize);
  for (size_t wIdx = 0; wIdx < wCoords.size(); wIdx += 4) {
    wIndex.add(wCoords[wIdx], wCoords[wIdx + 1], wCoords[wIdx + 2], wCoords[wIdx + 3]);
  }
  wIndex.finish();

  auto wEndTime = std::chrono::high_resolution_clock::now();
  std::cout << "index " << wNumItems << " rectangles: " << std::chrono::duration_cast<std::chrono::milliseconds>((wEndTime - wStartTime)).count() << "ms" << std::endl;
  std::cout << "index size: " << wIndex.data().size() << std::endl;

  benchSearch(wIndex, wBoxes100, wNumTests, 10);
  benchSearch(wIndex, wBoxes10, wNumTests, 1);
  benchSearch(wIndex, wBoxes1, wNumTests, 0.01);

  benchNeighbors(wIndex, wCoords, wNumTests, 100);
  benchNeighbors(wIndex, wCoords, 1, wNumItems);
  benchNeighbors(wIndex, wCoords, wNumItems / 10, 1);
}
