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

#include <assert.h>
#include <iostream>
#include <vector>

static const std::vector<double> gData
{
  8, 62, 11, 66, 57, 17, 57, 19, 76, 26, 79, 29, 36, 56, 38, 56, 92, 77, 96, 80, 87, 70, 90, 74,
  43, 41, 47, 43, 0, 58, 2, 62, 76, 86, 80, 89, 27, 13, 27, 15, 71, 63, 75, 67, 25, 2, 27, 2, 87,
  6, 88, 6, 22, 90, 23, 93, 22, 89, 22, 93, 57, 11, 61, 13, 61, 55, 63, 56, 17, 85, 21, 87, 33,
  43, 37, 43, 6, 1, 7, 3, 80, 87, 80, 87, 23, 50, 26, 52, 58, 89, 58, 89, 12, 30, 15, 34, 32, 58,
  36, 61, 41, 84, 44, 87, 44, 18, 44, 19, 13, 63, 15, 67, 52, 70, 54, 74, 57, 59, 58, 59, 17, 90,
  20, 92, 48, 53, 52, 56, 92, 68, 92, 72, 26, 52, 30, 52, 56, 23, 57, 26, 88, 48, 88, 48, 66, 13,
  67, 15, 7, 82, 8, 86, 46, 68, 50, 68, 37, 33, 38, 36, 6, 15, 8, 18, 85, 36, 89, 38, 82, 45, 84,
  48, 12, 2, 16, 3, 26, 15, 26, 16, 55, 23, 59, 26, 76, 37, 79, 39, 86, 74, 90, 77, 16, 75, 18,
  78, 44, 18, 45, 21, 52, 67, 54, 71, 59, 78, 62, 78, 24, 5, 24, 8, 64, 80, 64, 83, 66, 55, 70,
  55, 0, 17, 2, 19, 15, 71, 18, 74, 87, 57, 87, 59, 6, 34, 7, 37, 34, 30, 37, 32, 51, 19, 53, 19,
  72, 51, 73, 55, 29, 45, 30, 45, 94, 94, 96, 95, 7, 22, 11, 24, 86, 45, 87, 48, 33, 62, 34, 65,
  18, 10, 21, 14, 64, 66, 67, 67, 64, 25, 65, 28, 27, 4, 31, 6, 84, 4, 85, 5, 48, 80, 50, 81, 1,
  61, 3, 61, 71, 89, 74, 92, 40, 42, 43, 43, 27, 64, 28, 66, 46, 26, 50, 26, 53, 83, 57, 87, 14,
  75, 15, 79, 31, 45, 34, 45, 89, 84, 92, 88, 84, 51, 85, 53, 67, 87, 67, 89, 39, 26, 43, 27, 47,
  61, 47, 63, 23, 49, 25, 53, 12, 3, 14, 5, 16, 50, 19, 53, 63, 80, 64, 84, 22, 63, 22, 64, 26,
  66, 29, 66, 2, 15, 3, 15, 74, 77, 77, 79, 64, 11, 68, 11, 38, 4, 39, 8, 83, 73, 87, 77, 85, 52,
  89, 56, 74, 60, 76, 63, 62, 66, 65, 67
};

static const std::vector<uint8_t> gFlatbush
{
  251, 56, 16, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 64, 0, 0, 0, 0, 0, 0, 79, 64, 0,
  0, 0, 0, 0, 0, 38, 64, 0, 0, 0, 0, 0, 128, 80, 64, 0, 0, 0, 0, 0, 128, 76, 64, 0,
  0, 0, 0, 0, 0, 49, 64, 0, 0, 0, 0, 0, 128, 76, 64, 0, 0, 0, 0, 0, 0, 51, 64, 0,
  0, 0, 0, 0, 0, 83, 64, 0, 0, 0, 0, 0, 0, 58, 64, 0, 0, 0, 0, 0, 192, 83, 64, 0,
  0, 0, 0, 0, 0, 61, 64, 0, 0, 0, 0, 0, 0, 66, 64, 0, 0, 0, 0, 0, 0, 76, 64, 0,
  0, 0, 0, 0, 0, 67, 64, 0, 0, 0, 0, 0, 0, 76, 64, 0, 0, 0, 0, 0, 0, 87, 64, 0,
  0, 0, 0, 0, 64, 83, 64, 0, 0, 0, 0, 0, 0, 88, 64, 0, 0, 0, 0, 0, 0, 84, 64, 0,
  0, 0, 0, 0, 192, 85, 64, 0, 0, 0, 0, 0, 128, 81, 64, 0, 0, 0, 0, 0, 128, 86, 64, 0,
  0, 0, 0, 0, 128, 82, 64, 0, 0, 0, 0, 0, 128, 69, 64, 0, 0, 0, 0, 0, 128, 68, 64, 0,
  0, 0, 0, 0, 128, 71, 64, 0, 0, 0, 0, 0, 128, 69, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 77, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 79, 64, 0,
  0, 0, 0, 0, 0, 83, 64, 0, 0, 0, 0, 0, 128, 85, 64, 0, 0, 0, 0, 0, 0, 84, 64, 0,
  0, 0, 0, 0, 64, 86, 64, 0, 0, 0, 0, 0, 0, 59, 64, 0, 0, 0, 0, 0, 0, 42, 64, 0,
  0, 0, 0, 0, 0, 59, 64, 0, 0, 0, 0, 0, 0, 46, 64, 0, 0, 0, 0, 0, 192, 81, 64, 0,
  0, 0, 0, 0, 128, 79, 64, 0, 0, 0, 0, 0, 192, 82, 64, 0, 0, 0, 0, 0, 192, 80, 64, 0,
  0, 0, 0, 0, 0, 57, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 59, 64, 0,
  0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 192, 85, 64, 0, 0, 0, 0, 0, 0, 24, 64, 0,
  0, 0, 0, 0, 0, 86, 64, 0, 0, 0, 0, 0, 0, 24, 64, 0, 0, 0, 0, 0, 0, 54, 64, 0,
  0, 0, 0, 0, 128, 86, 64, 0, 0, 0, 0, 0, 0, 55, 64, 0, 0, 0, 0, 0, 64, 87, 64, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 88, 64, 0,
  0, 0, 0, 0, 64, 87, 64, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8,
  0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 0, 0
};

flatbush::Flatbush<double> createIndex()
{
  auto wNumItems = gData.size() / 4;
  flatbush::FlatbushBuilder<double> wBuilder;

  for (size_t wIdx = 0; wIdx < gData.size(); wIdx += 4)
  {
    wBuilder.add({ gData[wIdx], gData[wIdx + 1], gData[wIdx + 2], gData[wIdx + 3] });
  }
  auto wIndex = wBuilder.finish();

  assert(wIndex.numItems() == wNumItems);
  assert(wIndex.nodeSize() == flatbush::gDefaultNodeSize);

  return wIndex;
}

flatbush::Flatbush<double> createSmallIndex(uint32_t iNumItems, uint16_t iNodeSize)
{
  flatbush::FlatbushBuilder<double> wBuilder(iNumItems, iNodeSize);

  auto wSize = static_cast<size_t>(iNumItems) * 4;
  for (size_t wIdx = 0; wIdx < wSize; wIdx += 4)
  {
    wBuilder.add({ gData[wIdx], gData[wIdx + 1], gData[wIdx + 2], gData[wIdx + 3] });
  }
  auto wIndex = wBuilder.finish();

  assert(wIndex.numItems() == iNumItems);
  assert(wIndex.nodeSize() == iNodeSize);

  return wIndex;
}

void indexBunchOfRectangles()
{
  std::cout << "indexes a bunch of rectangles" << std::endl;
  auto wIndex = createIndex();
  assert(wIndex.boxSize() + wIndex.indexSize() == 540);

  auto wData = wIndex.data();
  auto wBoxes = reinterpret_cast<const double*>(&wData[flatbush::gHeaderByteSize]);
  size_t wBoxLen = wIndex.boxSize();
  assert(wBoxes[wBoxLen - 4] == 0);
  assert(wBoxes[wBoxLen - 3] == 1);
  assert(wBoxes[wBoxLen - 2] == 96);
  assert(wBoxes[wBoxLen - 1] == 95);

  auto wIndices = reinterpret_cast<const uint16_t*>(&wBoxes[wBoxLen]);
  assert(wIndices[wBoxLen / 4 - 1] == 400);
}

void skipSortingLessThanNodeSizeRectangles()
{
  std::cout << "skips sorting less than nodeSize number of rectangles" << std::endl;
  uint32_t wNumItems = 14;
  uint16_t wNodeSize = 16;
  auto wIndex = createSmallIndex(wNumItems, wNodeSize);

  // compute expected root box extents
  auto wRootMinX = std::numeric_limits<double>::max();
  auto wRootMinY = std::numeric_limits<double>::max();
  auto wRootMaxX = std::numeric_limits<double>::lowest();
  auto wRootMaxY = std::numeric_limits<double>::lowest();

  auto wSize = static_cast<size_t>(wNumItems);
  for (size_t wIdx = 0; wIdx < wSize * 4; wIdx += 4)
  {
    if (gData[wIdx]     < wRootMinX) wRootMinX = gData[wIdx];
    if (gData[wIdx + 1] < wRootMinY) wRootMinY = gData[wIdx + 1];
    if (gData[wIdx + 2] > wRootMaxX) wRootMaxX = gData[wIdx + 2];
    if (gData[wIdx + 3] > wRootMaxY) wRootMaxY = gData[wIdx + 3];
  }

  auto wData = wIndex.data();
  auto wBoxes = reinterpret_cast<const double*>(&wData[flatbush::gHeaderByteSize]);
  size_t wBoxLen = wIndex.boxSize();

  auto wIndices = reinterpret_cast<const uint16_t*>(&wBoxes[wBoxLen]);
  // sort should be skipped, ordered progressing indices expected
  for (size_t wIdx = 0; wIdx < wSize; ++wIdx)
  {
    assert(wIndices[wIdx] == wIdx);
  }
  assert(wIndices[wSize] == 0);

  assert(wBoxLen == (wSize + 1) * 4);
  assert(wBoxes[wBoxLen - 4] == wRootMinX);
  assert(wBoxes[wBoxLen - 3] == wRootMinY);
  assert(wBoxes[wBoxLen - 2] == wRootMaxX);
  assert(wBoxes[wBoxLen - 1] == wRootMaxY);
}

void performBoxSearch()
{
  std::cout << "performs bbox search" << std::endl;
  auto wIndex = createIndex();
  flatbush::Box<double> box{40, 40, 60, 60};
  auto wIds = wIndex.search(box);
  std::vector<double> wExpected = { 57, 59, 58, 59, 48, 53, 52, 56, 40, 42, 43, 43, 43, 41, 47, 43 };
  std::vector<double> wResults;

  for (size_t wIdx = 0; wIdx < wIds.size(); ++wIdx)
  {
    wResults.push_back(gData[4 * wIds[wIdx]]);
    wResults.push_back(gData[4 * wIds[wIdx] + 1]);
    wResults.push_back(gData[4 * wIds[wIdx] + 2]);
    wResults.push_back(gData[4 * wIds[wIdx] + 3]);
  }

  assert(wExpected.size() == wResults.size());
  std::sort(wExpected.begin(), wExpected.end());
  std::sort(wResults.begin(), wResults.end());

  assert(std::equal(wExpected.data(), wExpected.data() + wExpected.size(), wResults.data()));
}

void reconstructIndexFromArrayBuffer()
{
  std::cout << "reconstructs an index from array buffer" << std::endl;
  auto wIndex = createIndex();
  auto wIndexBuffer = wIndex.data();
  auto wIndex2 = flatbush::FlatbushBuilder<double>::from(wIndexBuffer.data());
  auto wIndex2Buffer = wIndex2.data();

  assert(wIndexBuffer.size() == wIndex2Buffer.size());

  assert(std::equal(wIndexBuffer.data(), wIndexBuffer.data() + wIndexBuffer.size(), wIndex2Buffer.data()));
}

void doesNotFreezeOnZeroNumItems()
{
  std::cout << "does not freeze on numItems = 0" << std::endl;
  bool wIsThrown = false;

  try
  {
    flatbush::FlatbushBuilder<double> wBuilder;
    wBuilder.finish();
  }
  catch (const std::invalid_argument& iError)
  {
    wIsThrown = (std::string("Unpexpected numItems value: 0.").compare(iError.what()) == 0);
  }

  assert(wIsThrown);
}

void performNeighborsQuery()
{
  std::cout << "performs a k-nearest-neighbors query" << std::endl;
  auto wIndex = createIndex();
  auto wIds = wIndex.neighbors({ 50, 50 }, 3);
  std::vector<size_t> wExpected = { 31, 6, 75 };

  assert(wExpected.size() == wIds.size());
  std::sort(wExpected.begin(), wExpected.end());
  std::sort(wIds.begin(), wIds.end());

  assert(std::equal(wExpected.data(), wExpected.data() + wExpected.size(), wIds.data()));
}

void neighborsQueryMaxDistance()
{
  std::cout << "k-nearest-neighbors query accepts maxDistance" << std::endl;
  auto wIndex = createIndex();
  auto wIds = wIndex.neighbors({ 50, 50 }, flatbush::gMaxUint32, 12);
  std::vector<size_t> wExpected = { 6, 29, 31, 75, 85 };

  assert(wExpected.size() == wIds.size());
  std::sort(wExpected.begin(), wExpected.end());
  std::sort(wIds.begin(), wIds.end());

  assert(std::equal(wExpected.data(), wExpected.data() + wExpected.size(), wIds.data()));
}

void neighborsQueryFilterFunc()
{
  std::cout << "k-nearest-neighbors query accepts filterFn" << std::endl;
  auto wIndex = createIndex();
  auto wIds = wIndex.neighbors({ 50, 50 }, 6, flatbush::gMaxDouble, [](size_t iValue) -> bool { return iValue % 2 == 0; });
  std::vector<size_t> wExpected = { 6, 16, 18, 24, 54, 80 };

  assert(wExpected.size() == wIds.size());
  std::sort(wExpected.begin(), wExpected.end());
  std::sort(wIds.begin(), wIds.end());

  assert(std::equal(wExpected.data(), wExpected.data() + wExpected.size(), wIds.data()));
}

void returnIndexOfNewlyAddedRectangle()
{
  std::cout << "returns index of newly-added rectangle" << std::endl;
  flatbush::FlatbushBuilder<double> wBuilder;

  for (size_t wIdx = 0; wIdx < 5; ++wIdx) {
    assert(wIdx == wBuilder.add({ gData[wIdx], gData[wIdx + 1], gData[wIdx + 2], gData[wIdx + 3] }));
  }
}

void searchQueryFilterFunc()
{
  std::cout << "bbox search query accepts filterFn" << std::endl;
  auto wIndex = createIndex();
  auto wIds = wIndex.search({ 40, 40, 60, 60 }, [](size_t iValue) -> bool { return iValue % 2 == 0; });

  assert(wIds.size() == 1);
  assert(wIds.front() == 6);
}

void reconstructIndexFromJSArrayBuffer()
{
  std::cout << "reconstructs an index from JS array buffer" << std::endl;
  auto wIndex = flatbush::FlatbushBuilder<double>::from(gFlatbush.data());
  auto wIndexBuffer = wIndex.data();

  assert(wIndexBuffer.size() == gFlatbush.size());

  assert(std::equal(wIndexBuffer.data(), wIndexBuffer.data() + wIndexBuffer.size(), gFlatbush.data()));
}

void wrongTemplateType()
{
  std::cout << "throws an error if creating instance of unsupported template type" << std::endl;
  bool wIsThrown = false;

  try
  {
    flatbush::FlatbushBuilder<std::string> wBuilder;
  }
  catch (const std::runtime_error& iError)
  {
    wIsThrown = (std::string("Unexpected typed array class. Expecting non 64-bit integral or floating point.").compare(iError.what()) == 0);
  }

  assert(wIsThrown);
}

void fromNull()
{
  std::cout << "throws an error if no data passed to from()" << std::endl;
  bool wIsThrown = false;

  try
  {
    flatbush::FlatbushBuilder<double>::from(nullptr);
  }
  catch (const std::invalid_argument& iError)
  {
    wIsThrown = (std::string("Data is incomplete or missing.").compare(iError.what()) == 0);
  }

  assert(wIsThrown);
}

void fromWrongMagic()
{
  std::cout << "throws an error if magic field is invalid" << std::endl;
  bool wIsThrown = false;

  try
  {
    flatbush::FlatbushBuilder<double>::from(std::vector<uint8_t>{ 0xf1 }.data());
  }
  catch (const std::invalid_argument& iError)
  {
    wIsThrown = (std::string("Data does not appear to be in a Flatbush format.").compare(iError.what()) == 0);
  }

  assert(wIsThrown);
}

void fromWrongVersion()
{
  std::cout << "throws an error on version mismatch" << std::endl;
  bool wIsThrown = false;

  try
  {
    flatbush::FlatbushBuilder<double>::from(std::vector<uint8_t>{ 0xfb, 2 << 4 }.data());
  }
  catch (const std::invalid_argument& iError)
  {
    wIsThrown = (std::string("Got v2 data when expected v" + std::to_string(flatbush::gVersion) + ".").compare(iError.what()) == 0);
  }

  assert(wIsThrown);
}

void fromWrongEncodedType()
{
  std::cout << "throws an error reconstructing a type distinct than template type" << std::endl;
  bool wIsThrown = false;

  try
  {
    flatbush::FlatbushBuilder<int>::from(gFlatbush.data());
  }
  catch (const std::runtime_error& iError)
  {
    wIsThrown = (std::string("Expected type is double, but got template type int32_t").compare(iError.what()) == 0);
  }

  assert(wIsThrown);
}

void searchQuerySinglePointSmallNumItems()
{
  std::cout << "bbox search query single point (same min/max) with numitems < nodesize" << std::endl;

  flatbush::FlatbushBuilder<int> wBuilder;
  wBuilder.add({ 0, 0, 0, 0 });
  auto wIndex = wBuilder.finish();

  assert(wIndex.numItems() == 1);
  assert(wIndex.nodeSize() == flatbush::gDefaultNodeSize);

  auto wIds = wIndex.search({ 0, 0, 0, 0 });
  assert(wIds.size() == 1);
  assert(wIds.front() == 0);
}

void searchQuerySinglePointLargeNumItems()
{
  std::cout << "bbox search query single point (same min/max) with numitems > nodesize" << std::endl;
  uint32_t wNumItems = 5;
  uint16_t wNodeSize = 4;

  flatbush::FlatbushBuilder<int> wBuilder(wNumItems, wNodeSize);
  wBuilder.add({ 0, 0, 0, 0 });
  wBuilder.add({ 0, 1, 0, 1 });
  wBuilder.add({ 1, 0, 1, 0 });
  wBuilder.add({ 1, 1, 1, 1 });
  wBuilder.add({ 1, 2, 3, 4 });
  auto wIndex = wBuilder.finish();

  assert(wIndex.numItems() == wNumItems);
  assert(wIndex.nodeSize() == wNodeSize);

  auto wIds = wIndex.search({ 0, 0, 0, 0 });
  assert(wIds.size() == 1);
  assert(wIds.front() == 0);
}

void searchQueryMultiPointSmallNumItems()
{
  std::cout << "bbox search query multiple points (same min/max) with numitems < nodesize" << std::endl;
  uint32_t wNumItems = 5;

  flatbush::FlatbushBuilder<int> wBuilder;
  wBuilder.add({ 0, 0, 0, 0 });
  wBuilder.add({ 0, 1, 0, 1 });
  wBuilder.add({ 1, 0, 1, 0 });
  wBuilder.add({ 1, 1, 1, 1 });
  wBuilder.add({ 1, 2, 3, 4 });
  auto wIndex = wBuilder.finish();

  assert(wIndex.numItems() == wNumItems);
  assert(wIndex.nodeSize() == flatbush::gDefaultNodeSize);

  auto wIds = wIndex.search({ 0, 0, 1, 1 });
  assert(wIds.size() == 4);
  assert(wIds.front() == 0);
  assert(wIds.back() == 3);
}

void searchQueryMultiPointLargeNumItems()
{
  std::cout << "bbox search query multiple points (same min/max) with numitems > nodesize" << std::endl;
  uint32_t wNumItems = 9;
  uint16_t wNodeSize = 4;

  flatbush::FlatbushBuilder<int> wBuilder(wNumItems, wNodeSize);
  wBuilder.add({ 0, 0, 0, 0 });
  wBuilder.add({ 0, 1, 0, 1 });
  wBuilder.add({ 1, 0, 1, 0 });
  wBuilder.add({ 1, 1, 1, 1 });
  wBuilder.add({ 1, 2, 3, 4 });
  wBuilder.add({ 5, 6, 7, 8 });
  wBuilder.add({ 1, 3, 5, 7 });
  wBuilder.add({ 2, 4, 6, 8 });
  wBuilder.add({ 9, 9, 9, 9 });
  auto wIndex = wBuilder.finish();

  assert(wIndex.numItems() == wNumItems);
  assert(wIndex.nodeSize() == wNodeSize);

  auto wIds = wIndex.search({ 0, 0, 1, 1 });
  assert(wIds.size() == 4);
  assert(wIds.front() == 0);
  assert(wIds.back() == 1);
}

void clearAndReuseBuilder()
{
  std::cout << "clear and reuse builder" << std::endl;

  flatbush::FlatbushBuilder<double> wBuilder;

  for (size_t wIdx = 0; wIdx < gData.size(); wIdx += 4)
  {
    wBuilder.add({ gData[wIdx], gData[wIdx + 1], gData[wIdx + 2], gData[wIdx + 3] });
  }

  auto wIndex = wBuilder.finish();
  wBuilder.add({ 1, 2, 3, 4 });
  auto wIndex2 = wBuilder.finish();

  assert(wIndex2.numItems() == wIndex.numItems() + 1);
  assert(wIndex2.nodeSize() == wIndex.nodeSize());

  wBuilder.clear();
  wBuilder.add({ 1, 2, 3, 4 });
  auto wIndex3 = wBuilder.finish();

  assert(wIndex3.numItems() == 1);
  assert(wIndex3.nodeSize() == wIndex2.nodeSize());
}

void testOneMillionItems()
{
  std::cout << "test adding one million items" << std::endl;

  flatbush::FlatbushBuilder<uint32_t> wBuilder;
  uint32_t wNumItems = 1000000;

  for (uint32_t wIdx = 0; wIdx < wNumItems; ++wIdx)
  {
    wBuilder.add({ wIdx, wIdx, wIdx, wIdx });
  }

  auto wIndex = wBuilder.finish();
  assert(wIndex.numItems() == wNumItems);
  assert(wIndex.nodeSize() == flatbush::gDefaultNodeSize);

  auto wIds = wIndex.search({ 0, 0, 0, 0 });
  assert(wIds.size() == 1);
  assert(wIds.front() == 0);

  auto wIds2 = wIndex.search({ 0, 0, wNumItems, wNumItems });
  assert(wIds2.size() == wNumItems);
}


int main(int argc, char** argv)
{
  indexBunchOfRectangles();
  skipSortingLessThanNodeSizeRectangles();
  performBoxSearch();
  reconstructIndexFromArrayBuffer();
  doesNotFreezeOnZeroNumItems();
  returnIndexOfNewlyAddedRectangle();
  performNeighborsQuery();
  neighborsQueryMaxDistance();
  neighborsQueryFilterFunc();
  searchQueryFilterFunc();
  reconstructIndexFromJSArrayBuffer();
  wrongTemplateType();
  fromNull();
  fromWrongMagic();
  fromWrongVersion();
  fromWrongEncodedType();
  searchQuerySinglePointSmallNumItems();
  searchQuerySinglePointLargeNumItems();
  searchQueryMultiPointSmallNumItems();
  searchQueryMultiPointLargeNumItems();
  clearAndReuseBuilder();
  testOneMillionItems();

  return EXIT_SUCCESS;
}
