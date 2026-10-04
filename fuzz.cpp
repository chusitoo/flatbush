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
#include <initializer_list>
#include <string>
#include <tuple>
#include <unordered_map>
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
    wBuilder.add({ coord, coord, coord, coord });
  }
  auto wIndex = wBuilder.finish();

  return wIndex;
}

auto serializedBytesDomain() { return fuzztest::StringOf(fuzztest::Arbitrary<char>()).WithMaxSize(64UL * 1024UL); }

template <typename ArrayType>
std::vector<std::tuple<std::string>> serializedSeeds() {
  std::vector<std::tuple<std::string>> wSeeds { { std::string {} } };
  for (const auto wNumItems : { 1U, 6U, 33U }) {
    const auto wIndex = createIndex<ArrayType>(wNumItems, 2);
    std::string wBytes(flatbush::detail::bit_cast<const char*>(wIndex.data().data()), wIndex.data().size());
    wSeeds.emplace_back(wBytes);
    wBytes.pop_back();
    wSeeds.emplace_back(std::move(wBytes));
  }
  return wSeeds;
}

template <typename ArrayType>
flatbush::Flatbush<ArrayType> createSearchIndex() {
  flatbush::FlatbushBuilder<ArrayType> wBuilder;

  wBuilder.add(
      { static_cast<ArrayType>(42), static_cast<ArrayType>(0), static_cast<ArrayType>(42), static_cast<ArrayType>(0) });
  auto wIndex = wBuilder.finish();

  return wIndex;
}

// =============================================================================
// FUZZ_TEST: FuzzFrom - Tests deserialization from binary data
// =============================================================================

template <typename ArrayType>
void FuzzFromTemplate(const std::string& data) {
  const uint8_t* iData = flatbush::detail::bit_cast<const uint8_t*>(data.data());
  size_t iSize = data.size();

  try {
    auto wIndex = flatbush::FlatbushBuilder<ArrayType>::from(iData, iSize);
    uint16_t wNodeSize;
    uint32_t wNumItems;
    std::memcpy(&wNodeSize, iData + 2, sizeof(wNodeSize));
    std::memcpy(&wNumItems, iData + 4, sizeof(wNumItems));

    ASSERT_EQ(wIndex.data().size(), iSize);
    ASSERT_EQ(wIndex.nodeSize(), wNodeSize);
    ASSERT_EQ(wIndex.numItems(), wNumItems);
    ASSERT_GE(wIndex.indexSize(), wNumItems);

    const auto wLowest = std::numeric_limits<ArrayType>::lowest();
    const auto wHighest = std::numeric_limits<ArrayType>::max();
    const auto wResults = wIndex.search({ wLowest, wLowest, wHighest, wHighest }, {}, 32);
    ASSERT_LE(wResults.size(), std::min<size_t>(32, wNumItems));
    for (const auto wId : wResults) ASSERT_LT(wId, wNumItems);

    const auto wPartialResults = wIndex.search({ 0, 0, 1, 1 }, {}, 32);
    ASSERT_LE(wPartialResults.size(), std::min<size_t>(32, wNumItems));
    for (const auto wId : wPartialResults) ASSERT_LT(wId, wNumItems);

    const auto wNeighbors = wIndex.neighbors({ 0, 0 }, 32);
    ASSERT_LE(wNeighbors.size(), std::min<size_t>(32, wNumItems));
    for (const auto wId : wNeighbors) ASSERT_LT(wId, wNumItems);
  } catch (const std::invalid_argument&) {
    return;
  }
}

void FuzzFromInt8(const std::string& data) { FuzzFromTemplate<int8_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromInt8).WithDomains(serializedBytesDomain()).WithSeeds(serializedSeeds<int8_t>);

void FuzzFromUInt8(const std::string& data) { FuzzFromTemplate<uint8_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromUInt8).WithDomains(serializedBytesDomain()).WithSeeds(serializedSeeds<uint8_t>);

void FuzzFromInt16(const std::string& data) { FuzzFromTemplate<int16_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromInt16).WithDomains(serializedBytesDomain()).WithSeeds(serializedSeeds<int16_t>);

void FuzzFromUInt16(const std::string& data) { FuzzFromTemplate<uint16_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromUInt16).WithDomains(serializedBytesDomain()).WithSeeds(serializedSeeds<uint16_t>);

void FuzzFromInt32(const std::string& data) { FuzzFromTemplate<int32_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromInt32).WithDomains(serializedBytesDomain()).WithSeeds(serializedSeeds<int32_t>);

void FuzzFromUInt32(const std::string& data) { FuzzFromTemplate<uint32_t>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromUInt32).WithDomains(serializedBytesDomain()).WithSeeds(serializedSeeds<uint32_t>);

void FuzzFromFloat(const std::string& data) { FuzzFromTemplate<float>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromFloat).WithDomains(serializedBytesDomain()).WithSeeds(serializedSeeds<float>);

void FuzzFromDouble(const std::string& data) { FuzzFromTemplate<double>(data); }
FUZZ_TEST(FlatbushFuzzTest, FuzzFromDouble).WithDomains(serializedBytesDomain()).WithSeeds(serializedSeeds<double>);

// =============================================================================
// FUZZ_TEST: FuzzSearch - Tests spatial search functionality
// =============================================================================

template <typename ArrayType>
void FuzzSearchTemplate(ArrayType minX, ArrayType minY, ArrayType maxX, ArrayType maxY) {
  auto wIndex = createSearchIndex<ArrayType>();
  auto wResult = wIndex.search({ minX, minY, maxX, maxY });

  if (minX <= static_cast<ArrayType>(42) && maxX >= static_cast<ArrayType>(42) && minY <= static_cast<ArrayType>(0) &&
      maxY >= static_cast<ArrayType>(0)) {
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

  const flatbush::Point<ArrayType> wPoint { iX, iY };
  const auto wResult = wIndex.neighbors(wPoint, iMaxResults, iMaxDistance);
  const auto wDistance = std::pow(wX - 42, 2.0) + std::pow(wY, 2.0);

  if (iMaxResults > 0 && iMaxDistance >= 0.0 && !std::isnan(wMaxDistSquared) && wDistance <= wMaxDistSquared) {
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

template <typename ArrayType>
flatbush::Box<ArrayType> orderedBox(const flatbush::Box<ArrayType>& iBox) {
  return { std::min(iBox.mMinX, iBox.mMaxX),
           std::min(iBox.mMinY, iBox.mMaxY),
           std::max(iBox.mMinX, iBox.mMaxX),
           std::max(iBox.mMinY, iBox.mMaxY) };
}

template <typename ArrayType>
auto indexDomains() {
  const auto wCoordinate = fuzztest::InRange<ArrayType>(static_cast<ArrayType>(std::is_signed<ArrayType>::value ? -100
                                                                                                                : 0),
                                                        static_cast<ArrayType>(100));
  const auto wBox = fuzztest::StructOf<flatbush::Box<ArrayType>>(wCoordinate, wCoordinate, wCoordinate, wCoordinate);
  return fuzztest::TupleOf(fuzztest::VectorOf(wBox).WithMinSize(1).WithMaxSize(8193),
                           fuzztest::OneOf(fuzztest::InRange<uint16_t>(0, 32), fuzztest::Just(flatbush::gMaxNodeSize)),
                           wBox,
                           fuzztest::Arbitrary<size_t>(),
                           fuzztest::Arbitrary<double>(),
                           fuzztest::Arbitrary<bool>());
}

template <typename ArrayType>
using IndexSeed = std::
    tuple<std::vector<flatbush::Box<ArrayType>>, uint16_t, flatbush::Box<ArrayType>, size_t, double, bool>;

template <typename ArrayType>
std::vector<IndexSeed<ArrayType>> indexSeeds() {
  const std::vector<flatbush::Box<ArrayType>> wOverlaps {
    { 0, 0, 0, 0 }, { 0, 0, 4, 4 }, { 4, 4, 0, 0 }, { 8, 8, 8, 8 }, { 0, 0, 0, 0 }
  };
  std::vector<IndexSeed<ArrayType>> wSeeds { { wOverlaps, 2, { 0, 0, 4, 4 }, 3, flatbush::gMaxDistance, false },
                                             { wOverlaps, 2, { 0, 0, 8, 8 }, flatbush::gMaxResults, 0.0, true },
                                             { wOverlaps, 0, { 1, 1, 3, 3 }, 0, 1.0, false } };
  for (const auto wNumItems : { 1UL, 17UL, 512UL, 513UL, 8192UL, 8193UL }) {
    if (wNumItems > 513UL && !std::is_same<ArrayType, double>::value) continue;
    std::vector<flatbush::Box<ArrayType>> wBoxes;
    wBoxes.reserve(wNumItems);
    for (size_t wId = 0UL; wId < wNumItems; ++wId) {
      const auto wCoordinate = static_cast<ArrayType>((wId * 37UL) % 101UL);
      wBoxes.push_back({ wCoordinate, wCoordinate, wCoordinate, wCoordinate });
    }
    wSeeds.emplace_back(std::move(wBoxes), 2, flatbush::Box<ArrayType> { 0, 0, 50, 50 }, 7, 60.0, false);
  }
  return wSeeds;
}

template <typename ArrayType>
void checkIndexSearch(std::initializer_list<const flatbush::Flatbush<ArrayType>*> iIndexes,
                      const std::vector<flatbush::Box<ArrayType>>& iBoxes,
                      const flatbush::Box<ArrayType>& iQuery,
                      size_t iMaxResults,
                      bool iEvenOnly) {
  std::vector<size_t> wExpected;
  for (size_t wId = 0UL; wId < iBoxes.size(); ++wId) {
    const auto wBox = orderedBox(iBoxes[wId]);
    const auto wAccept = !iEvenOnly || wId % 2UL == 0UL;
    if (wAccept && wBox.mMinX <= iQuery.mMaxX && wBox.mMaxX >= iQuery.mMinX && wBox.mMinY <= iQuery.mMaxY &&
        wBox.mMaxY >= iQuery.mMinY) {
      wExpected.push_back(wId);
    }
  }

  const auto wEvenFilter = [](size_t iId, const flatbush::Box<ArrayType>&) {
    return iId % 2UL == 0UL;
  };
  for (const auto* wIndex : iIndexes) {
    auto wResults = iEvenOnly ? wIndex->search(iQuery, wEvenFilter, iMaxResults)
                              : wIndex->search(iQuery, {}, iMaxResults);
    ASSERT_EQ(wResults.size(), std::min(iMaxResults, wExpected.size()));
    std::sort(wResults.begin(), wResults.end());
    ASSERT_TRUE(std::includes(wExpected.begin(), wExpected.end(), wResults.begin(), wResults.end()));
  }
}

template <typename ArrayType>
void checkIndexNeighbors(std::initializer_list<const flatbush::Flatbush<ArrayType>*> iIndexes,
                         const std::vector<flatbush::Box<ArrayType>>& iBoxes,
                         const flatbush::Point<ArrayType>& iPoint,
                         size_t iMaxResults,
                         double iMaxDistance,
                         bool iEvenOnly) {
  const auto wPointX = static_cast<double>(iPoint.mX);
  const auto wPointY = static_cast<double>(iPoint.mY);
  const auto wThreshold = iMaxDistance * iMaxDistance;
  std::unordered_map<size_t, double> wCandidates;
  std::vector<double> wExpectedDistances;

  for (size_t wId = 0UL; wId < iBoxes.size(); ++wId) {
    const auto wBox = static_cast<flatbush::Box<double>>(orderedBox(iBoxes[wId]));
    const auto wDeltaX = wPointX - std::clamp(wPointX, wBox.mMinX, wBox.mMaxX);
    const auto wDeltaY = wPointY - std::clamp(wPointY, wBox.mMinY, wBox.mMaxY);
    const auto wDistance = wDeltaX * wDeltaX + wDeltaY * wDeltaY;
    const auto wAccept = !iEvenOnly || wId % 2UL == 0UL;
    if (wAccept && iMaxDistance >= 0.0 && wDistance <= wThreshold) {
      wCandidates.emplace(wId, wDistance);
      wExpectedDistances.push_back(wDistance);
    }
  }

  std::sort(wExpectedDistances.begin(), wExpectedDistances.end());
  wExpectedDistances.resize(std::min(iMaxResults, wExpectedDistances.size()));
  const auto wEvenFilter = [](size_t iId, const flatbush::Box<ArrayType>&) {
    return iId % 2UL == 0UL;
  };
  for (const auto* wIndex : iIndexes) {
    const auto wNeighbors = iEvenOnly ? wIndex->neighbors(iPoint, iMaxResults, iMaxDistance, wEvenFilter)
                                      : wIndex->neighbors(iPoint, iMaxResults, iMaxDistance);
    ASSERT_EQ(wNeighbors.size(), wExpectedDistances.size());
    auto wRemaining = wCandidates;
    for (size_t wRank = 0UL; wRank < wNeighbors.size(); ++wRank) {
      const auto wCandidate = wRemaining.find(wNeighbors[wRank]);
      ASSERT_NE(wCandidate, wRemaining.end());
      ASSERT_DOUBLE_EQ(wCandidate->second, wExpectedDistances[wRank]);
      wRemaining.erase(wCandidate);
    }
  }
}

template <typename ArrayType>
void FuzzIndexTemplate(const std::vector<flatbush::Box<ArrayType>>& iBoxes,
                       uint16_t iNodeSize,
                       const flatbush::Box<ArrayType>& iQuery,
                       size_t iMaxResults,
                       double iMaxDistance,
                       bool iEvenOnly) {
  flatbush::FlatbushBuilder<ArrayType> wBuilder(iBoxes.size(), iNodeSize);
  for (const auto& wBox : iBoxes) {
    wBuilder.add(orderedBox(wBox));
  }

  const auto wIndex = wBuilder.finish();
  const auto wCopy = flatbush::FlatbushBuilder<ArrayType>::from(wIndex.data().data(), wIndex.data().size());
  const auto wView = flatbush::FlatbushBuilder<ArrayType>::fromView(wIndex.data());
  const auto wIndexes = { &wIndex, &wCopy, &wView };
  ASSERT_NO_FATAL_FAILURE(checkIndexSearch(wIndexes, iBoxes, orderedBox(iQuery), iMaxResults, iEvenOnly));
  ASSERT_NO_FATAL_FAILURE(
      checkIndexNeighbors(wIndexes, iBoxes, { iQuery.mMinX, iQuery.mMinY }, iMaxResults, iMaxDistance, iEvenOnly));
}

void FuzzIndexInt8(const std::vector<flatbush::Box<int8_t>>& iBoxes,
                   uint16_t iNodeSize,
                   const flatbush::Box<int8_t>& iQuery,
                   size_t iMaxResults,
                   double iMaxDistance,
                   bool iEvenOnly) {
  FuzzIndexTemplate(iBoxes, iNodeSize, iQuery, iMaxResults, iMaxDistance, iEvenOnly);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzIndexInt8).WithDomains(indexDomains<int8_t>()).WithSeeds(indexSeeds<int8_t>);

void FuzzIndexUInt8(const std::vector<flatbush::Box<uint8_t>>& iBoxes,
                    uint16_t iNodeSize,
                    const flatbush::Box<uint8_t>& iQuery,
                    size_t iMaxResults,
                    double iMaxDistance,
                    bool iEvenOnly) {
  FuzzIndexTemplate(iBoxes, iNodeSize, iQuery, iMaxResults, iMaxDistance, iEvenOnly);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzIndexUInt8).WithDomains(indexDomains<uint8_t>()).WithSeeds(indexSeeds<uint8_t>);

void FuzzIndexInt16(const std::vector<flatbush::Box<int16_t>>& iBoxes,
                    uint16_t iNodeSize,
                    const flatbush::Box<int16_t>& iQuery,
                    size_t iMaxResults,
                    double iMaxDistance,
                    bool iEvenOnly) {
  FuzzIndexTemplate(iBoxes, iNodeSize, iQuery, iMaxResults, iMaxDistance, iEvenOnly);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzIndexInt16).WithDomains(indexDomains<int16_t>()).WithSeeds(indexSeeds<int16_t>);

void FuzzIndexUInt16(const std::vector<flatbush::Box<uint16_t>>& iBoxes,
                     uint16_t iNodeSize,
                     const flatbush::Box<uint16_t>& iQuery,
                     size_t iMaxResults,
                     double iMaxDistance,
                     bool iEvenOnly) {
  FuzzIndexTemplate(iBoxes, iNodeSize, iQuery, iMaxResults, iMaxDistance, iEvenOnly);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzIndexUInt16).WithDomains(indexDomains<uint16_t>()).WithSeeds(indexSeeds<uint16_t>);

void FuzzIndexInt32(const std::vector<flatbush::Box<int32_t>>& iBoxes,
                    uint16_t iNodeSize,
                    const flatbush::Box<int32_t>& iQuery,
                    size_t iMaxResults,
                    double iMaxDistance,
                    bool iEvenOnly) {
  FuzzIndexTemplate(iBoxes, iNodeSize, iQuery, iMaxResults, iMaxDistance, iEvenOnly);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzIndexInt32).WithDomains(indexDomains<int32_t>()).WithSeeds(indexSeeds<int32_t>);

void FuzzIndexUInt32(const std::vector<flatbush::Box<uint32_t>>& iBoxes,
                     uint16_t iNodeSize,
                     const flatbush::Box<uint32_t>& iQuery,
                     size_t iMaxResults,
                     double iMaxDistance,
                     bool iEvenOnly) {
  FuzzIndexTemplate(iBoxes, iNodeSize, iQuery, iMaxResults, iMaxDistance, iEvenOnly);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzIndexUInt32).WithDomains(indexDomains<uint32_t>()).WithSeeds(indexSeeds<uint32_t>);

void FuzzIndexFloat(const std::vector<flatbush::Box<float>>& iBoxes,
                    uint16_t iNodeSize,
                    const flatbush::Box<float>& iQuery,
                    size_t iMaxResults,
                    double iMaxDistance,
                    bool iEvenOnly) {
  FuzzIndexTemplate(iBoxes, iNodeSize, iQuery, iMaxResults, iMaxDistance, iEvenOnly);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzIndexFloat).WithDomains(indexDomains<float>()).WithSeeds(indexSeeds<float>);

void FuzzIndexDouble(const std::vector<flatbush::Box<double>>& iBoxes,
                     uint16_t iNodeSize,
                     const flatbush::Box<double>& iQuery,
                     size_t iMaxResults,
                     double iMaxDistance,
                     bool iEvenOnly) {
  FuzzIndexTemplate(iBoxes, iNodeSize, iQuery, iMaxResults, iMaxDistance, iEvenOnly);
}
FUZZ_TEST(FlatbushFuzzTest, FuzzIndexDouble).WithDomains(indexDomains<double>()).WithSeeds(indexSeeds<double>);
