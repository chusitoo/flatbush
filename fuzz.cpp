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

#include <fuzztest/fuzztest.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "flatbush.h"

// =============================================================================
// Helper functions and utilities
// =============================================================================

template <typename ArrayType>
flatbush::Flatbush<ArrayType> createIndex(uint32_t iNumItems, uint16_t iNodeSize) {
  flatbush::FlatbushBuilder<ArrayType> wBuilder(iNumItems, iNodeSize);

  auto wSize = static_cast<size_t>(iNumItems);
  for (size_t wIdx = 0; wIdx < wSize; ++wIdx) {
    auto coord = static_cast<ArrayType>(wIdx);
    wBuilder.add({coord, coord, coord, coord});
  }
  auto wIndex = wBuilder.finish();

  return wIndex;
}

std::vector<size_t> calculateNumNodesPerLevel(uint32_t iNumItems, uint32_t iNodeSize) {
  size_t wCount = iNumItems;
  size_t wNumNodes = iNumItems;
  std::vector<size_t> wLevelBounds{wNumNodes};

  do {
    wCount = (wCount + iNodeSize - 1) / iNodeSize;
    wNumNodes += wCount;
    wLevelBounds.push_back(wNumNodes);
  } while (wCount > 1);

  return wLevelBounds;
}

template <typename ArrayType>
flatbush::Flatbush<ArrayType> createSearchIndex() {
  flatbush::FlatbushBuilder<ArrayType> wBuilder;

  wBuilder.add({static_cast<ArrayType>(42),
                static_cast<ArrayType>(0),
                static_cast<ArrayType>(42),
                static_cast<ArrayType>(0)});
  auto wIndex = wBuilder.finish();

  return wIndex;
}

// =============================================================================
// FUZZ_TEST: FuzzFrom - Tests deserialization from binary data
// =============================================================================

template <typename ArrayType>
void FuzzFromTemplate(const std::string& data) {
  const uint8_t* iData = reinterpret_cast<const uint8_t*>(data.data());
  size_t iSize = data.size();

  if (iSize < flatbush::gHeaderByteSize) return;
  if (iData[0] != flatbush::gValidityFlag) return;
  if ((iData[1] >> 4) != flatbush::gVersion) return;
  if ((iData[1] & 0x0f) != flatbush::detail::arrayTypeIndex<ArrayType>()) return;

  // Use memcpy to safely read unaligned data
  uint16_t wNodeSize;
  std::memcpy(&wNodeSize, &iData[2], sizeof(uint16_t));
  if (wNodeSize < 2) return;

  uint32_t wNumItems;
  std::memcpy(&wNumItems, &iData[4], sizeof(uint32_t));

  const auto& wLevelBounds = calculateNumNodesPerLevel(wNumItems, wNodeSize);
  const auto wNumNodes = wLevelBounds.empty() ? wNumItems : wLevelBounds.back();
  const auto wIndicesByteSize =
      wNumNodes * ((wNumNodes > flatbush::gMaxNumNodes) ? sizeof(uint32_t) : sizeof(uint16_t));
  const auto wNodesByteSize = wNumNodes * sizeof(flatbush::Box<ArrayType>);
  const auto wSize = flatbush::gHeaderByteSize + wNodesByteSize + wIndicesByteSize;
  if (wSize != iSize) return;

  auto wIndex = flatbush::FlatbushBuilder<ArrayType>::from(iData, iSize);

  ASSERT_EQ(wIndex.data().size(), iSize);
  ASSERT_EQ(wIndex.nodeSize(), wNodeSize);
  ASSERT_EQ(wIndex.numItems(), wNumItems);
  ASSERT_EQ(wIndex.indexSize(), wNumNodes);
}

void FuzzFromInt8(const std::string& data) { FuzzFromTemplate<int8_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromInt8);

void FuzzFromUInt8(const std::string& data) { FuzzFromTemplate<uint8_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromUInt8);

void FuzzFromInt16(const std::string& data) { FuzzFromTemplate<int16_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromInt16);

void FuzzFromUInt16(const std::string& data) { FuzzFromTemplate<uint16_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromUInt16);

void FuzzFromInt32(const std::string& data) { FuzzFromTemplate<int32_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromInt32);

void FuzzFromUInt32(const std::string& data) { FuzzFromTemplate<uint32_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromUInt32);

void FuzzFromFloat(const std::string& data) { FuzzFromTemplate<float>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromFloat);

void FuzzFromDouble(const std::string& data) { FuzzFromTemplate<double>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromDouble);

// =============================================================================
// FUZZ_TEST: FuzzSearch - Tests spatial search functionality
// =============================================================================

template <typename ArrayType>
void FuzzSearchTemplate(ArrayType minX, ArrayType minY, ArrayType maxX, ArrayType maxY) {
  auto wIndex = createSearchIndex<ArrayType>();
  auto wResult = wIndex.search({minX, minY, maxX, maxY});

  if (minX <= static_cast<ArrayType>(42) && maxX >= static_cast<ArrayType>(42) &&
      minY <= static_cast<ArrayType>(0) && maxY >= static_cast<ArrayType>(0)) {
    ASSERT_EQ(wResult.size(), 1);
  } else {
    ASSERT_EQ(wResult.size(), 0);
  }
}

void FuzzSearchInt8(int8_t minX, int8_t minY, int8_t maxX, int8_t maxY) {
  FuzzSearchTemplate<int8_t>(minX, minY, maxX, maxY);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzSearchInt8);

void FuzzSearchUInt8(uint8_t minX, uint8_t minY, uint8_t maxX, uint8_t maxY) {
  FuzzSearchTemplate<uint8_t>(minX, minY, maxX, maxY);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzSearchUInt8);

void FuzzSearchInt16(int16_t minX, int16_t minY, int16_t maxX, int16_t maxY) {
  FuzzSearchTemplate<int16_t>(minX, minY, maxX, maxY);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzSearchInt16);

void FuzzSearchUInt16(uint16_t minX, uint16_t minY, uint16_t maxX, uint16_t maxY) {
  FuzzSearchTemplate<uint16_t>(minX, minY, maxX, maxY);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzSearchUInt16);

void FuzzSearchInt32(int32_t minX, int32_t minY, int32_t maxX, int32_t maxY) {
  FuzzSearchTemplate<int32_t>(minX, minY, maxX, maxY);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzSearchInt32);

void FuzzSearchUInt32(uint32_t minX, uint32_t minY, uint32_t maxX, uint32_t maxY) {
  FuzzSearchTemplate<uint32_t>(minX, minY, maxX, maxY);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzSearchUInt32);

void FuzzSearchFloat(float minX, float minY, float maxX, float maxY) {
  FuzzSearchTemplate<float>(minX, minY, maxX, maxY);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzSearchFloat);

void FuzzSearchDouble(double minX, double minY, double maxX, double maxY) {
  FuzzSearchTemplate<double>(minX, minY, maxX, maxY);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzSearchDouble);

// =============================================================================
// FUZZ_TEST: FuzzNeighbors - Tests nearest neighbor search functionality
// =============================================================================

template <typename ArrayType>
void FuzzNeighborsTemplate(ArrayType iX, ArrayType iY, size_t iMaxResults, double iMaxDistance) {
  auto wIndex = createSearchIndex<ArrayType>();
  const auto wX = static_cast<double>(iX);
  const auto wY = static_cast<double>(iY);
  const auto wMaxDistSquared = iMaxDistance * iMaxDistance;

  const flatbush::Point<ArrayType> wPoint{iX, iY};
  const auto wResult = wIndex.neighbors(wPoint, iMaxResults, iMaxDistance);
  const auto wDistance = std::pow(wX - 42, 2.0) + std::pow(wY, 2.0);

  if (iMaxResults > 0 && iMaxDistance >= 0.0 && std::isnormal(wMaxDistSquared) &&
      wDistance <= wMaxDistSquared) {
    ASSERT_EQ(wResult.size(), 1);
  } else {
    ASSERT_EQ(wResult.size(), 0);
  }
}

void FuzzNeighborsInt8(int8_t iX, int8_t iY, size_t iMaxResults, double iMaxDistance) {
  FuzzNeighborsTemplate<int8_t>(iX, iY, iMaxResults, iMaxDistance);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzNeighborsInt8);

void FuzzNeighborsUInt8(uint8_t iX, uint8_t iY, size_t iMaxResults, double iMaxDistance) {
  FuzzNeighborsTemplate<uint8_t>(iX, iY, iMaxResults, iMaxDistance);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzNeighborsUInt8);

void FuzzNeighborsInt16(int16_t iX, int16_t iY, size_t iMaxResults, double iMaxDistance) {
  FuzzNeighborsTemplate<int16_t>(iX, iY, iMaxResults, iMaxDistance);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzNeighborsInt16);

void FuzzNeighborsUInt16(uint16_t iX, uint16_t iY, size_t iMaxResults, double iMaxDistance) {
  FuzzNeighborsTemplate<uint16_t>(iX, iY, iMaxResults, iMaxDistance);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzNeighborsUInt16);

void FuzzNeighborsInt32(int32_t iX, int32_t iY, size_t iMaxResults, double iMaxDistance) {
  FuzzNeighborsTemplate<int32_t>(iX, iY, iMaxResults, iMaxDistance);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzNeighborsInt32);

void FuzzNeighborsUInt32(uint32_t iX, uint32_t iY, size_t iMaxResults, double iMaxDistance) {
  FuzzNeighborsTemplate<uint32_t>(iX, iY, iMaxResults, iMaxDistance);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzNeighborsUInt32);

void FuzzNeighborsFloat(float iX, float iY, size_t iMaxResults, double iMaxDistance) {
  FuzzNeighborsTemplate<float>(iX, iY, iMaxResults, iMaxDistance);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzNeighborsFloat);

void FuzzNeighborsDouble(double iX, double iY, size_t iMaxResults, double iMaxDistance) {
  FuzzNeighborsTemplate<double>(iX, iY, iMaxResults, iMaxDistance);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzNeighborsDouble);
