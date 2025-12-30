/*
MIT License

Copyright (c) 2025 Alex Emirov

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

#include <gtest/gtest.h>

#include <vector>

#include "flatbush.h"

static constexpr std::array<double, 400> gData{
    8,  62, 11, 66, 57, 17, 57, 19, 76, 26, 79, 29, 36, 56, 38, 56, 92, 77, 96, 80, 87, 70, 90, 74,
    43, 41, 47, 43, 0,  58, 2,  62, 76, 86, 80, 89, 27, 13, 27, 15, 71, 63, 75, 67, 25, 2,  27, 2,
    87, 6,  88, 6,  22, 90, 23, 93, 22, 89, 22, 93, 57, 11, 61, 13, 61, 55, 63, 56, 17, 85, 21, 87,
    33, 43, 37, 43, 6,  1,  7,  3,  80, 87, 80, 87, 23, 50, 26, 52, 58, 89, 58, 89, 12, 30, 15, 34,
    32, 58, 36, 61, 41, 84, 44, 87, 44, 18, 44, 19, 13, 63, 15, 67, 52, 70, 54, 74, 57, 59, 58, 59,
    17, 90, 20, 92, 48, 53, 52, 56, 92, 68, 92, 72, 26, 52, 30, 52, 56, 23, 57, 26, 88, 48, 88, 48,
    66, 13, 67, 15, 7,  82, 8,  86, 46, 68, 50, 68, 37, 33, 38, 36, 6,  15, 8,  18, 85, 36, 89, 38,
    82, 45, 84, 48, 12, 2,  16, 3,  26, 15, 26, 16, 55, 23, 59, 26, 76, 37, 79, 39, 86, 74, 90, 77,
    16, 75, 18, 78, 44, 18, 45, 21, 52, 67, 54, 71, 59, 78, 62, 78, 24, 5,  24, 8,  64, 80, 64, 83,
    66, 55, 70, 55, 0,  17, 2,  19, 15, 71, 18, 74, 87, 57, 87, 59, 6,  34, 7,  37, 34, 30, 37, 32,
    51, 19, 53, 19, 72, 51, 73, 55, 29, 45, 30, 45, 94, 94, 96, 95, 7,  22, 11, 24, 86, 45, 87, 48,
    33, 62, 34, 65, 18, 10, 21, 14, 64, 66, 67, 67, 64, 25, 65, 28, 27, 4,  31, 6,  84, 4,  85, 5,
    48, 80, 50, 81, 1,  61, 3,  61, 71, 89, 74, 92, 40, 42, 43, 43, 27, 64, 28, 66, 46, 26, 50, 26,
    53, 83, 57, 87, 14, 75, 15, 79, 31, 45, 34, 45, 89, 84, 92, 88, 84, 51, 85, 53, 67, 87, 67, 89,
    39, 26, 43, 27, 47, 61, 47, 63, 23, 49, 25, 53, 12, 3,  14, 5,  16, 50, 19, 53, 63, 80, 64, 84,
    22, 63, 22, 64, 26, 66, 29, 66, 2,  15, 3,  15, 74, 77, 77, 79, 64, 11, 68, 11, 38, 4,  39, 8,
    83, 73, 87, 77, 85, 52, 89, 56, 74, 60, 76, 63, 62, 66, 65, 67};

static const std::array<uint8_t, 518> gFlatbush{
    251, 56, 16, 0, 14, 0,   0,  0,  0,  0, 0,  0, 0, 0,   32, 64, 0, 0, 0, 0, 0, 0,   79, 64,
    0,   0,  0,  0, 0,  0,   38, 64, 0,  0, 0,  0, 0, 128, 80, 64, 0, 0, 0, 0, 0, 128, 76, 64,
    0,   0,  0,  0, 0,  0,   49, 64, 0,  0, 0,  0, 0, 128, 76, 64, 0, 0, 0, 0, 0, 0,   51, 64,
    0,   0,  0,  0, 0,  0,   83, 64, 0,  0, 0,  0, 0, 0,   58, 64, 0, 0, 0, 0, 0, 192, 83, 64,
    0,   0,  0,  0, 0,  0,   61, 64, 0,  0, 0,  0, 0, 0,   66, 64, 0, 0, 0, 0, 0, 0,   76, 64,
    0,   0,  0,  0, 0,  0,   67, 64, 0,  0, 0,  0, 0, 0,   76, 64, 0, 0, 0, 0, 0, 0,   87, 64,
    0,   0,  0,  0, 0,  64,  83, 64, 0,  0, 0,  0, 0, 0,   88, 64, 0, 0, 0, 0, 0, 0,   84, 64,
    0,   0,  0,  0, 0,  192, 85, 64, 0,  0, 0,  0, 0, 128, 81, 64, 0, 0, 0, 0, 0, 128, 86, 64,
    0,   0,  0,  0, 0,  128, 82, 64, 0,  0, 0,  0, 0, 128, 69, 64, 0, 0, 0, 0, 0, 128, 68, 64,
    0,   0,  0,  0, 0,  128, 71, 64, 0,  0, 0,  0, 0, 128, 69, 64, 0, 0, 0, 0, 0, 0,   0,  0,
    0,   0,  0,  0, 0,  0,   77, 64, 0,  0, 0,  0, 0, 0,   0,  64, 0, 0, 0, 0, 0, 0,   79, 64,
    0,   0,  0,  0, 0,  0,   83, 64, 0,  0, 0,  0, 0, 128, 85, 64, 0, 0, 0, 0, 0, 0,   84, 64,
    0,   0,  0,  0, 0,  64,  86, 64, 0,  0, 0,  0, 0, 0,   59, 64, 0, 0, 0, 0, 0, 0,   42, 64,
    0,   0,  0,  0, 0,  0,   59, 64, 0,  0, 0,  0, 0, 0,   46, 64, 0, 0, 0, 0, 0, 192, 81, 64,
    0,   0,  0,  0, 0,  128, 79, 64, 0,  0, 0,  0, 0, 192, 82, 64, 0, 0, 0, 0, 0, 192, 80, 64,
    0,   0,  0,  0, 0,  0,   57, 64, 0,  0, 0,  0, 0, 0,   0,  64, 0, 0, 0, 0, 0, 0,   59, 64,
    0,   0,  0,  0, 0,  0,   0,  64, 0,  0, 0,  0, 0, 192, 85, 64, 0, 0, 0, 0, 0, 0,   24, 64,
    0,   0,  0,  0, 0,  0,   86, 64, 0,  0, 0,  0, 0, 0,   24, 64, 0, 0, 0, 0, 0, 0,   54, 64,
    0,   0,  0,  0, 0,  128, 86, 64, 0,  0, 0,  0, 0, 0,   55, 64, 0, 0, 0, 0, 0, 64,  87, 64,
    0,   0,  0,  0, 0,  0,   0,  0,  0,  0, 0,  0, 0, 0,   0,  64, 0, 0, 0, 0, 0, 0,   88, 64,
    0,   0,  0,  0, 0,  64,  87, 64, 0,  0, 1,  0, 2, 0,   3,  0,  4, 0, 5, 0, 6, 0,   7,  0,
    8,   0,  9,  0, 10, 0,   11, 0,  12, 0, 13, 0, 0, 0};

flatbush::Flatbush<double> createIndex() {
  auto wNumItems = gData.size() / 4;
  flatbush::FlatbushBuilder<double> wBuilder;

  for (size_t wIdx = 0; wIdx < gData.size(); wIdx += 4) {
    wBuilder.add({gData[wIdx], gData[wIdx + 1], gData[wIdx + 2], gData[wIdx + 3]});
  }
  auto wIndex = wBuilder.finish();

  EXPECT_EQ(wIndex.numItems(), wNumItems);
  EXPECT_EQ(wIndex.nodeSize(), flatbush::gDefaultNodeSize);

  return wIndex;
}

flatbush::Flatbush<double> createSmallIndex(uint32_t iNumItems, uint16_t iNodeSize) {
  flatbush::FlatbushBuilder<double> wBuilder(iNumItems, iNodeSize);

  auto wSize = static_cast<size_t>(iNumItems) * 4;
  for (size_t wIdx = 0; wIdx < wSize; wIdx += 4) {
    wBuilder.add({gData[wIdx], gData[wIdx + 1], gData[wIdx + 2], gData[wIdx + 3]});
  }
  auto wIndex = wBuilder.finish();

  EXPECT_EQ(wIndex.numItems(), iNumItems);
  EXPECT_EQ(wIndex.nodeSize(), iNodeSize);

  return wIndex;
}

TEST(FlatbushTest, IndexBunchOfRectangles) {
  auto wIndex = createIndex();
  EXPECT_EQ(wIndex.indexSize() * 4 + wIndex.indexSize(), 540);

  auto wData = wIndex.data();
  auto wBoxes = flatbush::detail::bit_cast<const double*>(&wData[flatbush::gHeaderByteSize]);
  size_t wBoxLen = wIndex.indexSize() * 4;
  EXPECT_EQ(wBoxes[wBoxLen - 4], 0);
  EXPECT_EQ(wBoxes[wBoxLen - 3], 1);
  EXPECT_EQ(wBoxes[wBoxLen - 2], 96);
  EXPECT_EQ(wBoxes[wBoxLen - 1], 95);

  auto wIndices = flatbush::detail::bit_cast<const uint16_t*>(&wBoxes[wBoxLen]);
  EXPECT_EQ(wIndices[wBoxLen / 4 - 1], 400);
}

TEST(FlatbushTest, SkipSortingLessThanNodeSizeRectangles) {
  uint32_t wNumItems = 14;
  uint16_t wNodeSize = 16;
  auto wIndex = createSmallIndex(wNumItems, wNodeSize);

  // compute expected root box extents
  auto wRootMinX = std::numeric_limits<double>::max();
  auto wRootMinY = std::numeric_limits<double>::max();
  auto wRootMaxX = std::numeric_limits<double>::lowest();
  auto wRootMaxY = std::numeric_limits<double>::lowest();

  auto wSize = static_cast<size_t>(wNumItems);
  for (size_t wIdx = 0; wIdx < wSize * 4; wIdx += 4) {
    if (gData[wIdx] < wRootMinX) wRootMinX = gData[wIdx];
    if (gData[wIdx + 1] < wRootMinY) wRootMinY = gData[wIdx + 1];
    if (gData[wIdx + 2] > wRootMaxX) wRootMaxX = gData[wIdx + 2];
    if (gData[wIdx + 3] > wRootMaxY) wRootMaxY = gData[wIdx + 3];
  }

  auto wData = wIndex.data();
  auto wBoxes = flatbush::detail::bit_cast<const double*>(&wData[flatbush::gHeaderByteSize]);
  size_t wBoxLen = wIndex.indexSize() * 4;

  auto wIndices = flatbush::detail::bit_cast<const uint16_t*>(&wBoxes[wBoxLen]);
  // sort should be skipped, ordered progressing indices expected
  for (size_t wIdx = 0; wIdx < wSize; ++wIdx) {
    EXPECT_EQ(wIndices[wIdx], wIdx);
  }
  EXPECT_EQ(wIndices[wSize], 0);

  EXPECT_EQ(wBoxLen, (wSize + 1) * 4);
  EXPECT_EQ(wBoxes[wBoxLen - 4], wRootMinX);
  EXPECT_EQ(wBoxes[wBoxLen - 3], wRootMinY);
  EXPECT_EQ(wBoxes[wBoxLen - 2], wRootMaxX);
  EXPECT_EQ(wBoxes[wBoxLen - 1], wRootMaxY);
}

TEST(FlatbushTest, PerformBoxSearch) {
  auto wIndex = createIndex();
  flatbush::Box<double> box{40, 40, 60, 60};
  auto wIds = wIndex.search(box);
  std::vector<double> wExpected = {57, 59, 58, 59, 48, 53, 52, 56, 40, 42, 43, 43, 43, 41, 47, 43};
  std::vector<double> wResults;

  for (const auto wId : wIds) {
    wResults.push_back(gData[4 * wId]);
    wResults.push_back(gData[4 * wId + 1]);
    wResults.push_back(gData[4 * wId + 2]);
    wResults.push_back(gData[4 * wId + 3]);
  }

  EXPECT_EQ(wExpected.size(), wResults.size());
  std::sort(wExpected.begin(), wExpected.end());
  std::sort(wResults.begin(), wResults.end());

  EXPECT_TRUE(std::equal(wExpected.begin(), wExpected.end(), wResults.begin()));
}

TEST(FlatbushTest, ReconstructIndexFromArrayBuffer) {
  auto wIndex = createIndex();
  auto wIndexBuffer = wIndex.data();
  auto wIndex2 = flatbush::FlatbushBuilder<double>::from(wIndexBuffer.data(), wIndexBuffer.size());
  auto wIndex2Buffer = wIndex2.data();

  EXPECT_EQ(wIndexBuffer.size(), wIndex2Buffer.size());

  EXPECT_TRUE(std::equal(wIndexBuffer.begin(), wIndexBuffer.end(), wIndex2Buffer.begin()));
}

TEST(FlatbushTest, DoesNotFreezeOnZeroNumItems) {
  EXPECT_THROW(
      {
        flatbush::FlatbushBuilder<double> wBuilder;
        wBuilder.finish();
      },
      std::invalid_argument);
}

TEST(FlatbushTest, PerformNeighborsQuery) {
  auto wIndex = createIndex();
  auto wIds = wIndex.neighbors({50, 50}, 3);
  std::vector<size_t> wExpected = {31, 6, 75};

  EXPECT_EQ(wExpected.size(), wIds.size());
  std::sort(wExpected.begin(), wExpected.end());
  std::sort(wIds.begin(), wIds.end());

  EXPECT_TRUE(std::equal(wExpected.begin(), wExpected.end(), wIds.begin()));
}

TEST(FlatbushTest, NeighborsQueryAllItems) {
  auto wIndex = createIndex();
  auto wIds = wIndex.neighbors({50, 50});

  EXPECT_EQ(wIds.size(), wIndex.numItems());
}

TEST(FlatbushTest, NeighborsQueryMaxDistance) {
  auto wIndex = createIndex();
  auto wIds = wIndex.neighbors({50, 50}, flatbush::gMaxResults, 12);
  std::vector<size_t> wExpected = {6, 29, 31, 75, 85};

  EXPECT_EQ(wExpected.size(), wIds.size());
  std::sort(wExpected.begin(), wExpected.end());
  std::sort(wIds.begin(), wIds.end());

  EXPECT_TRUE(std::equal(wExpected.begin(), wExpected.end(), wIds.begin()));
}

TEST(FlatbushTest, NeighborsQueryFilterFunc) {
  auto wIndex = createIndex();
  auto wIds = wIndex.neighbors(
      {50, 50}, 6, flatbush::gMaxDistance, [](size_t iValue, const flatbush::Box<double>&) {
        return iValue % 2 == 0;
      });
  std::vector<size_t> wExpected = {6, 16, 18, 24, 54, 80};

  EXPECT_EQ(wExpected.size(), wIds.size());
  std::sort(wExpected.begin(), wExpected.end());
  std::sort(wIds.begin(), wIds.end());

  EXPECT_TRUE(std::equal(wExpected.begin(), wExpected.end(), wIds.begin()));
}

TEST(FlatbushTest, ReturnIndexOfNewlyAddedRectangle) {
  flatbush::FlatbushBuilder<double> wBuilder;

  for (size_t wIdx = 0; wIdx < 5; ++wIdx) {
    EXPECT_EQ(wIdx, wBuilder.add({gData[wIdx], gData[wIdx + 1], gData[wIdx + 2], gData[wIdx + 3]}));
  }
}

TEST(FlatbushTest, SearchQueryFilterFunc) {
  auto wIndex = createIndex();
  auto wIds = wIndex.search({40, 40, 60, 60}, [](size_t iValue, const flatbush::Box<double>&) {
    return iValue % 2 == 0;
  });
  EXPECT_EQ(wIds.size(), 1);
  EXPECT_EQ(wIds.front(), 6);
}

TEST(FlatbushTest, ReconstructIndexFromJSArrayBuffer) {
  auto wIndex = flatbush::FlatbushBuilder<double>::from(gFlatbush.data(), gFlatbush.size());
  auto wIndexBuffer = wIndex.data();

  EXPECT_EQ(wIndexBuffer.size(), gFlatbush.size());

  EXPECT_TRUE(std::equal(wIndexBuffer.begin(), wIndexBuffer.end(), gFlatbush.begin()));
}

TEST(FlatbushTest, FromNull) {
  EXPECT_THROW(
      { flatbush::FlatbushBuilder<double>::from(nullptr, flatbush::gHeaderByteSize); },
      std::invalid_argument);
}

TEST(FlatbushTest, FromWrongMagic) {
  EXPECT_THROW(
      {
        flatbush::FlatbushBuilder<double>::from(std::vector<uint8_t>{0xf1}.data(),
                                                flatbush::gHeaderByteSize);
      },
      std::invalid_argument);
}

TEST(FlatbushTest, FromWrongVersion) {
  EXPECT_THROW(
      {
        flatbush::FlatbushBuilder<double>::from(std::vector<uint8_t>{0xfb, 2 << 4}.data(),
                                                flatbush::gHeaderByteSize);
      },
      std::invalid_argument);
}

TEST(FlatbushTest, FromWrongEncodedType) {
  EXPECT_THROW(
      { flatbush::FlatbushBuilder<int>::from(gFlatbush.data(), gFlatbush.size()); },
      std::invalid_argument);
}

TEST(FlatbushTest, FromInvalidHeaderSize) {
  EXPECT_THROW(
      { flatbush::FlatbushBuilder<double>::from(std::vector<uint8_t>{251, 56, 0, 0}.data(), 4); },
      std::invalid_argument);
}

TEST(FlatbushTest, FromInvalidNodeSize) {
  EXPECT_THROW(
      {
        flatbush::FlatbushBuilder<double>::from(
            std::vector<uint8_t>{251, 56, 0, 0, 0, 0, 0, 0}.data(), flatbush::gHeaderByteSize);
      },
      std::invalid_argument);
}

TEST(FlatbushTest, FromInvalidNumItems) {
  EXPECT_THROW(
      {
        flatbush::FlatbushBuilder<double>::from(
            std::vector<uint8_t>{251, 56, 16, 0, 14, 0, 0, 0}.data(), flatbush::gHeaderByteSize);
      },
      std::invalid_argument);
}

TEST(FlatbushTest, AdjustedNodeSize) {
  flatbush::FlatbushBuilder<int> wBuilder0(1, 0);
  wBuilder0.add({0, 0, 0, 0});
  auto wIndex0 = wBuilder0.finish();
  EXPECT_EQ(wIndex0.numItems(), 1);
  EXPECT_EQ(wIndex0.nodeSize(), 2);

  flatbush::FlatbushBuilder<int> wBuilder1(1, 1);
  wBuilder1.add({0, 0, 0, 0});
  auto wIndex1 = wBuilder1.finish();
  EXPECT_EQ(wIndex1.numItems(), 1);
  EXPECT_EQ(wIndex1.nodeSize(), 2);
}

TEST(FlatbushTest, SearchQuerySinglePointSmallNumItems) {
  flatbush::FlatbushBuilder<int> wBuilder;
  wBuilder.add({0, 0, 0, 0});
  auto wIndex = wBuilder.finish();

  EXPECT_EQ(wIndex.numItems(), 1);
  EXPECT_EQ(wIndex.nodeSize(), flatbush::gDefaultNodeSize);

  auto wIds = wIndex.search({0, 0, 0, 0});
  EXPECT_EQ(wIds.size(), 1);
  EXPECT_EQ(wIds.front(), 0);
}

TEST(FlatbushTest, SearchQuerySinglePointLargeNumItems) {
  uint32_t wNumItems = 5;
  uint16_t wNodeSize = 4;

  flatbush::FlatbushBuilder<int> wBuilder(wNumItems, wNodeSize);
  wBuilder.add({0, 0, 0, 0});
  wBuilder.add({0, 1, 0, 1});
  wBuilder.add({1, 0, 1, 0});
  wBuilder.add({1, 1, 1, 1});
  wBuilder.add({1, 2, 3, 4});
  auto wIndex = wBuilder.finish();

  EXPECT_EQ(wIndex.numItems(), wNumItems);
  EXPECT_EQ(wIndex.nodeSize(), wNodeSize);

  auto wIds = wIndex.search({0, 0, 0, 0});
  EXPECT_EQ(wIds.size(), 1);
  EXPECT_EQ(wIds.front(), 0);
}

TEST(FlatbushTest, SearchQueryMultiPointSmallNumItems) {
  uint32_t wNumItems = 5;

  flatbush::FlatbushBuilder<int> wBuilder;
  wBuilder.add({0, 0, 0, 0});
  wBuilder.add({0, 1, 0, 1});
  wBuilder.add({1, 0, 1, 0});
  wBuilder.add({1, 1, 1, 1});
  wBuilder.add({1, 2, 3, 4});
  auto wIndex = wBuilder.finish();

  EXPECT_EQ(wIndex.numItems(), wNumItems);
  EXPECT_EQ(wIndex.nodeSize(), flatbush::gDefaultNodeSize);

  auto wIds = wIndex.search({0, 0, 1, 1});
  EXPECT_EQ(wIds.size(), 4);
  EXPECT_EQ(wIds.front(), 0);
  EXPECT_EQ(wIds.back(), 3);
}

TEST(FlatbushTest, SearchQueryMultiPointLargeNumItems) {
  uint32_t wNumItems = 9;
  uint16_t wNodeSize = 4;

  flatbush::FlatbushBuilder<int> wBuilder(wNumItems, wNodeSize);
  wBuilder.add({0, 0, 0, 0});
  wBuilder.add({0, 1, 0, 1});
  wBuilder.add({1, 0, 1, 0});
  wBuilder.add({1, 1, 1, 1});
  wBuilder.add({1, 2, 3, 4});
  wBuilder.add({5, 6, 7, 8});
  wBuilder.add({1, 3, 5, 7});
  wBuilder.add({2, 4, 6, 8});
  wBuilder.add({9, 9, 9, 9});
  auto wIndex = wBuilder.finish();

  EXPECT_EQ(wIndex.numItems(), wNumItems);
  EXPECT_EQ(wIndex.nodeSize(), wNodeSize);

  auto wIds = wIndex.search({0, 0, 1, 1});
  EXPECT_EQ(wIds.size(), 4);
  EXPECT_EQ(wIds.front(), 0);
  EXPECT_EQ(wIds.back(), 1);
}

TEST(FlatbushTest, ClearAndReuseBuilder) {
  flatbush::FlatbushBuilder<double> wBuilder;

  for (size_t wIdx = 0; wIdx < gData.size(); wIdx += 4) {
    wBuilder.add({gData[wIdx], gData[wIdx + 1], gData[wIdx + 2], gData[wIdx + 3]});
  }

  auto wIndex = wBuilder.finish();
  wBuilder.add({1, 2, 3, 4});
  auto wIndex2 = wBuilder.finish();

  EXPECT_EQ(wIndex2.numItems(), wIndex.numItems() + 1);
  EXPECT_EQ(wIndex2.nodeSize(), wIndex.nodeSize());

  wBuilder.clear();
  wBuilder.add({1, 2, 3, 4});
  auto wIndex3 = wBuilder.finish();

  EXPECT_EQ(wIndex3.numItems(), 1);
  EXPECT_EQ(wIndex3.nodeSize(), wIndex2.nodeSize());
}

TEST(FlatbushTest, TestOneMillionItems) {
  flatbush::FlatbushBuilder<uint32_t> wBuilder;
  uint32_t wNumItems = 1000000;

  for (uint32_t wIdx = 0; wIdx < wNumItems; ++wIdx) {
    wBuilder.add({wIdx, wIdx, wIdx, wIdx});
  }

  auto wIndex = wBuilder.finish();
  EXPECT_EQ(wIndex.numItems(), wNumItems);
  EXPECT_EQ(wIndex.nodeSize(), flatbush::gDefaultNodeSize);

  auto wIds = wIndex.search({0, 0, 0, 0});
  EXPECT_EQ(wIds.size(), 1);
  EXPECT_EQ(wIds.front(), 0);

  auto wIds2 = wIndex.search({0, 0, wNumItems, wNumItems});
  EXPECT_EQ(wIds2.size(), wNumItems);
}

TEST(FlatbushTest, QuickSortImbalancedDataset) {
  static const auto linspace = [](double wStart, double wStop, uint32_t wNum) {
    const auto wStep = (wStop - wStart) / (wNum - 1);
    std::vector<double> wItems(wNum);
    for (uint32_t wIndex = 0; wIndex < wNum; ++wIndex) {
      wItems.at(wIndex) = wStart + wStep * static_cast<double>(wIndex);
    }
    return wItems;
  };

  EXPECT_NO_THROW({
    flatbush::FlatbushBuilder<double> wBuilder;
    uint32_t wNumItems = 15000;
    const auto& wItems = linspace(0, 1000, wNumItems);

    for (uint32_t wCount = 0; wCount < 10; ++wCount) {
      for (const auto wItem : wItems) {
        wBuilder.add({wItem, 0, wItem, 0});
      }
    }
    wBuilder.finish();
  });
}

TEST(FlatbushTest, QuickSortWorksOnDuplicates) {
  uint32_t wNumItems = 55000 + 5500 + 7700;
  flatbush::FlatbushBuilder<double> wBuilder(wNumItems);
  auto wX = 0.0;

  for (uint32_t wCount = 0; wCount < 55000; ++wCount, ++wX) {
    wBuilder.add({wX, 3.0, wX, 3.0});
  }

  for (uint32_t wCount = 0; wCount < 5500; ++wCount, ++wX) {
    wBuilder.add({wX, 4.0, wX, 4.0});
  }

  for (uint32_t wCount = 0; wCount < 7700; ++wCount, ++wX) {
    wBuilder.add({wX, 5.0, wX, 5.0});
  }

  const auto wIndex = wBuilder.finish();

  const auto wIds = wIndex.search({0.5, -1, 6.5, 1});
  EXPECT_EQ(wIds.size(), 0);

  const auto wIds2 = wIndex.search({55000, 4.0, 55000, 4.0});
  EXPECT_EQ(wIds2.size(), 1);
}

TEST(FlatbushTest, ReconstructIndexFromMovedVector) {
  auto wIndex = createIndex();
  auto wIndexBuffer = wIndex.data();
  auto wIndexVector = std::vector<uint8_t>{wIndexBuffer.begin(), wIndexBuffer.end()};
  auto wIndex2 = flatbush::FlatbushBuilder<double>::from(std::move(wIndexVector));
  auto wIndex2Buffer = wIndex2.data();

  EXPECT_EQ(wIndexBuffer.size(), wIndex2Buffer.size());

  EXPECT_TRUE(std::equal(wIndexBuffer.begin(), wIndexBuffer.end(), wIndex2Buffer.begin()));
}
