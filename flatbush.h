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

#ifndef FLATBUSH_H_
#define FLATBUSH_H_

#include <algorithm>    // for max, min, upper_bound
#include <array>        // for array
#include <cmath>        // for isnan
#include <cstdint>      // for uint32_t, uint8_t, uint16_t, int16_t, int32_t, int8_t
#include <cstring>      // for size_t, memcpy
#include <functional>   // for function
#include <limits>       // for numeric_limits
#include <queue>        // for queue, priority_queue
#include <stdexcept>    // for invalid_argument
#include <string>       // for operator+, to_string, allocator, basic_string, char_traits, string
#include <type_traits>  // for enable_if, is_same, false_type, integral_constant
#include <utility>      // for swap
#include <vector>       // for vector

#ifndef FLATBUSH_SPAN
#include <span>
namespace flatbush {
using std::span;
}
#else
namespace flatbush {

template <typename Type>
class span {
  Type* mPtr = nullptr;
  size_t mLen = 0;

 public:
  span() noexcept = default;
  span(Type* iPtr, size_t iLen) noexcept : mPtr{iPtr}, mLen{iLen} {}
  Type& operator[](size_t iIndex) noexcept { return mPtr[iIndex]; }
  Type const& operator[](size_t iIndex) const noexcept { return mPtr[iIndex]; }
  const Type* data() const noexcept { return mPtr; }
  size_t size() const noexcept { return mLen; }
  Type* begin() noexcept { return mPtr; }
  Type* end() noexcept { return mPtr + mLen; }
};

}  // namespace flatbush
#endif  // FLATBUSH_SPAN

namespace flatbush {

using FilterCb = std::function<bool(size_t)>;
constexpr auto gMaxHilbert = std::numeric_limits<uint16_t>::max();
constexpr auto gMaxDistance = std::numeric_limits<double>::max();
constexpr auto gMaxResults = std::numeric_limits<size_t>::max();
constexpr auto gInvalidArrayType = std::numeric_limits<uint8_t>::max();
constexpr uint16_t gMinNodeSize = 2;
constexpr uint16_t gMaxNodeSize = std::numeric_limits<uint16_t>::max();
constexpr size_t gMaxNumNodes = std::numeric_limits<uint16_t>::max() / 4U;
constexpr size_t gDefaultNodeSize = 16;
constexpr size_t gHeaderByteSize = 8;
constexpr uint8_t gValidityFlag = 0xfb;
constexpr uint8_t gVersion = 3;  // serialized format version

namespace detail {

// From https://www.boost.org/doc/libs/1_81_0/boost/core/bit.hpp (modified)
template <class To, class From>
To bit_cast(From const& from) {
  static_assert(sizeof(To) == sizeof(From), "Cannot cast types of different size");

  To to;
  std::memcpy(&to, &from, sizeof(To));
  return to;
}

// From https://github.com/rawrunprotected/hilbert_curves (public domain)
inline uint32_t Interleave(uint32_t x) {
  x = (x | (x << 8U)) & 0x00FF00FF;
  x = (x | (x << 4U)) & 0x0F0F0F0F;
  x = (x | (x << 2U)) & 0x33333333;
  x = (x | (x << 1U)) & 0x55555555;
  return x;
}

inline uint32_t HilbertXYToIndex(uint32_t x, uint32_t y) {
  // Initial prefix scan round, prime with x and y
  uint32_t a = x ^ y;
  uint32_t b = 0xFFFF ^ a;
  uint32_t c = 0xFFFF ^ (x | y);
  uint32_t d = x & (y ^ 0xFFFF);
  uint32_t A = a | (b >> 1U);
  uint32_t B = (a >> 1U) ^ a;
  uint32_t C = ((c >> 1U) ^ (b & (d >> 1U))) ^ c;
  uint32_t D = ((a & (c >> 1U)) ^ (d >> 1U)) ^ d;

  a = A;
  b = B;
  c = C;
  d = D;
  A = ((a & (a >> 2U)) ^ (b & (b >> 2U)));
  B = ((a & (b >> 2U)) ^ (b & ((a ^ b) >> 2U)));
  C ^= ((a & (c >> 2U)) ^ (b & (d >> 2U)));
  D ^= ((b & (c >> 2U)) ^ ((a ^ b) & (d >> 2U)));

  a = A;
  b = B;
  c = C;
  d = D;
  A = ((a & (a >> 4U)) ^ (b & (b >> 4U)));
  B = ((a & (b >> 4U)) ^ (b & ((a ^ b) >> 4U)));
  C ^= ((a & (c >> 4U)) ^ (b & (d >> 4U)));
  D ^= ((b & (c >> 4U)) ^ ((a ^ b) & (d >> 4U)));

  // Final round and projection
  a = A;
  b = B;
  c = C;
  d = D;
  C ^= ((a & (c >> 8U)) ^ (b & (d >> 8U)));
  D ^= ((b & (c >> 8U)) ^ ((a ^ b) & (d >> 8U)));

  // Undo transformation prefix scan
  a = C ^ (C >> 1U);
  b = D ^ (D >> 1U);

  // Recover index bits
  uint32_t i0 = x ^ y;
  uint32_t i1 = b | (0xFFFF ^ (i0 | a));

  return ((Interleave(i1) << 1U) | Interleave(i0));
}

// Template specialization for the supported array types
template <typename Type, typename...>
struct is_contained : std::false_type {};

template <typename Type, typename Head, typename... Tail>
struct is_contained<Type, Head, Tail...>
    : std::integral_constant<bool,
                             std::is_same<Type, Head>::value ||
                                 is_contained<Type, Tail...>::value> {};

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, int8_t>::value, uint8_t>::type
arrayTypeIndex() {
  return 0;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, uint8_t>::value, uint8_t>::type
arrayTypeIndex() {
  return 1;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, int16_t>::value, uint8_t>::type
arrayTypeIndex() {
  return 3;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, uint16_t>::value, uint8_t>::type
arrayTypeIndex() {
  return 4;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, int32_t>::value, uint8_t>::type
arrayTypeIndex() {
  return 5;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, uint32_t>::value, uint8_t>::type
arrayTypeIndex() {
  return 6;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, float>::value, uint8_t>::type
arrayTypeIndex() {
  return 7;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, double>::value, uint8_t>::type
arrayTypeIndex() {
  return 8;
}

template <typename ArrayType>
constexpr typename std::enable_if<
    !is_contained<ArrayType, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, float, double>::
        value,
    uint8_t>::type
arrayTypeIndex() {
  return gInvalidArrayType;
}

inline const std::string& arrayTypeName(size_t iIndex) {
  static const std::string kUnknownType{"unknown"};
  static const std::array<std::string, 9> kArrayTypeNames{"int8_t",
                                                          "uint8_t",
                                                          "uint8_t",
                                                          "int16_t",
                                                          "uint16_t",
                                                          "int32_t",
                                                          "uint32_t",
                                                          "float",
                                                          "double"};
  return iIndex < kArrayTypeNames.size() ? kArrayTypeNames[iIndex] : kUnknownType;
}
}  // namespace detail

template <typename ArrayType>
struct Box {
  ArrayType mMinX;
  ArrayType mMinY;
  ArrayType mMaxX;
  ArrayType mMaxY;
};

template <typename ArrayType>
struct Point {
  ArrayType mX;
  ArrayType mY;
};

template <class ArrayType>
class Flatbush;
template <class ArrayType>
class FlatbushBuilder {
 public:
  explicit FlatbushBuilder(size_t iNumItems = 10, uint16_t iNodeSize = gDefaultNodeSize)
      : mNodeSize(iNodeSize) {
    static_assert(detail::arrayTypeIndex<ArrayType>() != gInvalidArrayType,
                  "Unexpected typed array class. Expecting non 64-bit integral "
                  "or floating point.");

    mItems.reserve(iNumItems);
  }

  inline void clear() noexcept { mItems.clear(); };

  inline size_t add(const Box<ArrayType>& iBox) noexcept {
    mItems.push_back(iBox);
    return mItems.size() - 1UL;
  }

  Flatbush<ArrayType> finish() const;
  static Flatbush<ArrayType> from(const uint8_t* iData, size_t iSize);

 private:
  std::uint16_t mNodeSize;
  std::vector<Box<ArrayType>> mItems;
};

template <typename ArrayType>
Flatbush<ArrayType> FlatbushBuilder<ArrayType>::finish() const {
  if (mItems.empty()) {
    throw std::invalid_argument("No items have been added. Nothing to build.");
  }

  Flatbush<ArrayType> wIndex(uint32_t(mItems.size()), mNodeSize);
  for (const auto& wItem : mItems) {
    wIndex.add(wItem);
  }
  wIndex.finish();

  return wIndex;
}

template <typename ArrayType>
Flatbush<ArrayType> FlatbushBuilder<ArrayType>::from(const uint8_t* iData, size_t iSize) {
  static_assert(detail::arrayTypeIndex<ArrayType>() != gInvalidArrayType,
                "Unexpected typed array class. Expecting non 64-bit integral "
                "or floating point.");

  if (iSize < gHeaderByteSize) {
    throw std::invalid_argument("Data buffer size must be at least " +
                                std::to_string(gHeaderByteSize) + " bytes.");
  }

  if (iData == nullptr) {
    throw std::invalid_argument("Data is incomplete or missing.");
  }

  const auto wMagic = iData[0];
  if (wMagic != gValidityFlag) {
    throw std::invalid_argument("Data does not appear to be in a Flatbush format.");
  }

  const uint8_t wEncodedVersion = iData[1] >> 4U;
  if (wEncodedVersion != gVersion) {
    throw std::invalid_argument("Got v" + std::to_string(wEncodedVersion) +
                                " data when expected v" + std::to_string(gVersion) + ".");
  }

  constexpr auto wExpectedType = detail::arrayTypeIndex<ArrayType>();
  const uint8_t wEncodedType = iData[1] & 0x0fU;
  if (wExpectedType != wEncodedType) {
    throw std::invalid_argument("Expected type is " + detail::arrayTypeName(wEncodedType) +
                                ", but got template type " + detail::arrayTypeName(wExpectedType));
  }

  const auto wNodeSize = *detail::bit_cast<uint16_t*>(&iData[2]);
  if (wNodeSize < gMinNodeSize) {
    throw std::invalid_argument("Node size cannot be < " + std::to_string(gMinNodeSize) + ".");
  }

  auto wInstance = Flatbush<ArrayType>(iData, iSize);
  const auto wSize = wInstance.data().size();
  if (wSize != iSize) {
    throw std::invalid_argument("Num items dictates a total size of " + std::to_string(wSize) +
                                ", but got buffer size " + std::to_string(iSize) + ".");
  }

  return wInstance;
}

template <typename ArrayType>
class Flatbush {
 public:
  Flatbush(const Flatbush&) = delete;
  Flatbush& operator=(const Flatbush&) = delete;
  Flatbush(Flatbush&&) noexcept = default;
  Flatbush& operator=(Flatbush&&) noexcept = default;
  ~Flatbush() = default;

  std::vector<size_t> search(const Box<ArrayType>& iBounds,
                             const FilterCb& iFilterFn = nullptr) const noexcept;

  std::vector<size_t> neighbors(const Point<ArrayType>& iPoint,
                                size_t iMaxResults = gMaxResults,
                                double iMaxDistance = gMaxDistance,
                                const FilterCb& iFilterFn = nullptr) const noexcept;

  inline size_t nodeSize() const noexcept { return *detail::bit_cast<uint16_t*>(&mData[2]); };

  inline size_t numItems() const noexcept { return *detail::bit_cast<uint32_t*>(&mData[4]); };

  inline size_t indexSize() const noexcept { return mBoxes.size(); };

  inline span<const uint8_t> data() const noexcept { return {mData.data(), mData.capacity()}; };

  friend class FlatbushBuilder<ArrayType>;

 private:
  static constexpr ArrayType cMaxValue = std::numeric_limits<ArrayType>::max();
  static constexpr ArrayType cMinValue = std::numeric_limits<ArrayType>::lowest();

  static inline double axisDistance(ArrayType iValue, ArrayType iMin, ArrayType iMax) noexcept {
    return iValue < iMin ? iMin - iValue : std::max(iValue - iMax, 0.0);
  }

  static inline bool canDoSearch(const Box<ArrayType>& iBounds) {
#if defined(_WIN32) || defined(_WIN64)
    // On Windows, isnan throws on anything that is not float, double or long double
    const auto wIsNanBounds = (std::isnan(static_cast<double>(iBounds.mMinX)) ||
                               std::isnan(static_cast<double>(iBounds.mMinY)) ||
                               std::isnan(static_cast<double>(iBounds.mMaxX)) ||
                               std::isnan(static_cast<double>(iBounds.mMaxY)));
#else
    const auto wIsNanBounds = (std::isnan(iBounds.mMinX) || std::isnan(iBounds.mMinY) ||
                               std::isnan(iBounds.mMaxX) || std::isnan(iBounds.mMaxY));
#endif
    return !wIsNanBounds;
  }

  static inline bool canDoNeighbors(const Point<ArrayType>& iPoint,
                                    size_t iMaxResults,
                                    double iMaxDistance) {
#if defined(_WIN32) || defined(_WIN64)
    // On Windows, isnan throws on anything that is not float, double or long double
    const auto wIsNanPoint =
        (std::isnan(static_cast<double>(iPoint.mX)) || std::isnan(static_cast<double>(iPoint.mY)));
#else
    const auto wIsNanPoint = (std::isnan(iPoint.mX) || std::isnan(iPoint.mY));
#endif
    return !wIsNanPoint && !std::isnan(iMaxDistance) && iMaxDistance >= 0.0 && iMaxResults != 0UL;
  }

  explicit Flatbush(uint32_t iNumItems, uint16_t iNodeSize) noexcept;
  explicit Flatbush(const uint8_t* iData, size_t iSize) noexcept;

  size_t add(const Box<ArrayType>& iBox) noexcept;
  void finish() noexcept;
  void init(uint32_t iNumItems, uint32_t iNodeSize) noexcept;
  void sort(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight) noexcept;
  void swap(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight) noexcept;
  size_t upperBound(size_t iNodeIndex) const noexcept;

  struct IndexDistance {
    IndexDistance(size_t iId, ArrayType iDistance) noexcept : mId(iId), mDistance(iDistance) {}
    bool operator<(const IndexDistance& iOther) const { return iOther.mDistance < mDistance; }
    bool operator>(const IndexDistance& iOther) const { return iOther.mDistance > mDistance; }

    size_t mId;
    ArrayType mDistance;
  };

  // views
  std::vector<uint8_t> mData;
  span<Box<ArrayType>> mBoxes;
  span<uint16_t> mIndicesUint16;
  span<uint32_t> mIndicesUint32;
  // pick appropriate index view
  bool mIsWideIndex = false;
  // box stuff
  size_t mPosition = 0;
  std::vector<size_t> mLevelBounds;
  Box<ArrayType> mBounds;
};

template <typename ArrayType>
Flatbush<ArrayType>::Flatbush(uint32_t iNumItems, uint16_t iNodeSize) noexcept {
  iNodeSize = std::min(std::max(iNodeSize, gMinNodeSize), gMaxNodeSize);
  init(iNumItems, iNodeSize);

  mData.assign(mData.capacity(), 0U);
  mData[0] = gValidityFlag;
  mData[1] = (gVersion << 4U) + detail::arrayTypeIndex<ArrayType>();
  *detail::bit_cast<uint16_t*>(&mData[2]) = iNodeSize;
  *detail::bit_cast<uint32_t*>(&mData[4]) = iNumItems;
}

template <typename ArrayType>
Flatbush<ArrayType>::Flatbush(const uint8_t* iData, size_t iSize) noexcept {
  const auto wNodeSize = *detail::bit_cast<const uint16_t*>(&iData[2]);
  const auto wNumItems = *detail::bit_cast<const uint32_t*>(&iData[4]);
  init(wNumItems, wNodeSize);

  mData.insert(mData.begin(), iData, iData + iSize);
  mPosition = mLevelBounds.empty() ? 0UL : mLevelBounds.back();
  if (mPosition > 0UL) {
    mBounds = mBoxes[mPosition - 1UL];
  }
}

template <typename ArrayType>
void Flatbush<ArrayType>::init(uint32_t iNumItems, uint32_t iNodeSize) noexcept {
  mBounds = {cMaxValue, cMaxValue, cMinValue, cMinValue};

  // calculate the total number of nodes in the R-tree to allocate space for
  // and the index of each tree level (used in search later)
  size_t wCount = iNumItems;
  size_t wNumNodes = iNumItems;
  mLevelBounds.push_back(wNumNodes);

  do {
    wCount = (wCount + iNodeSize - 1UL) / iNodeSize;
    wNumNodes += wCount;
    mLevelBounds.push_back(wNumNodes);
  } while (wCount > 1UL);

  // Sizes
  mIsWideIndex = wNumNodes > gMaxNumNodes;
  const size_t wIndicesByteSize = wNumNodes * (mIsWideIndex ? sizeof(uint32_t) : sizeof(uint16_t));
  const size_t wNodesByteSize = wNumNodes * sizeof(Box<ArrayType>);
  const size_t wDataSize = gHeaderByteSize + wNodesByteSize + wIndicesByteSize;
  // Views
  mData.reserve(wDataSize);
  mBoxes = {detail::bit_cast<Box<ArrayType>*>(&mData[gHeaderByteSize]), wNumNodes};
  mIndicesUint16 = {detail::bit_cast<uint16_t*>(&mData[gHeaderByteSize + wNodesByteSize]),
                    wNumNodes};
  mIndicesUint32 = {detail::bit_cast<uint32_t*>(&mData[gHeaderByteSize + wNodesByteSize]),
                    wNumNodes};
}

template <typename ArrayType>
size_t Flatbush<ArrayType>::add(const Box<ArrayType>& iBox) noexcept {
  if (mIsWideIndex) {
    mIndicesUint32[mPosition] = uint32_t(mPosition);
  } else {
    mIndicesUint16[mPosition] = uint16_t(mPosition);
  }

  mBoxes[mPosition] = iBox;
  mBounds.mMinX = std::min(mBounds.mMinX, iBox.mMinX);
  mBounds.mMinY = std::min(mBounds.mMinY, iBox.mMinY);
  mBounds.mMaxX = std::max(mBounds.mMaxX, iBox.mMaxX);
  mBounds.mMaxY = std::max(mBounds.mMaxY, iBox.mMaxY);
  return mPosition++;
}

template <typename ArrayType>
void Flatbush<ArrayType>::finish() noexcept {
  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();

  if (wNumItems <= wNodeSize) {
    mBoxes[mPosition++] = mBounds;
    return;
  }

  std::vector<uint32_t> wHilbertValues(wNumItems);
  const auto wHilbertWidth = gMaxHilbert / (mBounds.mMaxX - mBounds.mMinX);
  const auto wHilbertHeight = gMaxHilbert / (mBounds.mMaxY - mBounds.mMinY);

  // map item centers into Hilbert coordinate space and calculate Hilbert values
  for (size_t wIdx = 0UL; wIdx < wNumItems; ++wIdx) {
    wHilbertValues.at(wIdx) = detail::HilbertXYToIndex(
        uint32_t(wHilbertWidth * ((mBoxes[wIdx].mMinX + mBoxes[wIdx].mMaxX) / 2 - mBounds.mMinX)),
        uint32_t(wHilbertHeight * ((mBoxes[wIdx].mMinY + mBoxes[wIdx].mMaxY) / 2 - mBounds.mMinY)));
  }

  // sort items by their Hilbert value (for packing later)
  sort(wHilbertValues, 0U, wNumItems - 1U);

  for (size_t wIdx = 0UL, wPosition = 0UL; wIdx < mLevelBounds.size() - 1UL; ++wIdx) {
    const auto wEnd = mLevelBounds[wIdx];

    // generate a parent node for each block of consecutive <nodeSize> nodes
    while (wPosition < wEnd) {
      const auto wNodeIndex = wPosition << 2U;  // for binary compatibility with JS
      auto wNodeBox = mBoxes[wPosition];

      // calculate bbox for the new node
      for (size_t wCount = 0; wCount < wNodeSize && wPosition < wEnd; ++wCount, ++wPosition) {
        wNodeBox.mMinX = std::min(wNodeBox.mMinX, mBoxes[wPosition].mMinX);
        wNodeBox.mMinY = std::min(wNodeBox.mMinY, mBoxes[wPosition].mMinY);
        wNodeBox.mMaxX = std::max(wNodeBox.mMaxX, mBoxes[wPosition].mMaxX);
        wNodeBox.mMaxY = std::max(wNodeBox.mMaxY, mBoxes[wPosition].mMaxY);
      }

      // add the new node to the tree data
      if (mIsWideIndex) {
        mIndicesUint32[mPosition] = uint32_t(wNodeIndex);
      } else {
        mIndicesUint16[mPosition] = uint16_t(wNodeIndex);
      }

      mBoxes[mPosition++] = wNodeBox;
    }
  }
}

// custom quicksort that partially sorts bbox data alongside the hilbert values
template <typename ArrayType>
void Flatbush<ArrayType>::sort(std::vector<uint32_t>& iValues,
                               size_t iLeft,
                               size_t iRight) noexcept {
  if (iLeft < iRight) {
    const auto wPivot = iValues.at((iLeft + iRight) >> 1U);
    auto wPivotLeft = iLeft - 1U;
    auto wPivotRight = iRight + 1U;

    while (true) {
      do {
        ++wPivotLeft;
      } while (iValues.at(wPivotLeft) < wPivot);

      do {
        --wPivotRight;
      } while (iValues.at(wPivotRight) > wPivot);

      if (wPivotLeft >= wPivotRight) {
        break;
      }

      swap(iValues, wPivotLeft, wPivotRight);
    }

    sort(iValues, iLeft, wPivotRight);
    sort(iValues, wPivotRight + 1U, iRight);
  }
}

// swap two values and two corresponding boxes
template <typename ArrayType>
void Flatbush<ArrayType>::swap(std::vector<uint32_t>& iValues,
                               size_t iLeft,
                               size_t iRight) noexcept {
  std::swap(iValues.at(iLeft), iValues.at(iRight));
  std::swap(mBoxes[iLeft], mBoxes[iRight]);

  if (mIsWideIndex) {
    std::swap(mIndicesUint32[iLeft], mIndicesUint32[iRight]);
  } else {
    std::swap(mIndicesUint16[iLeft], mIndicesUint16[iRight]);
  }
}

template <typename ArrayType>
size_t Flatbush<ArrayType>::upperBound(size_t iNodeIndex) const noexcept {
  const auto& wIt = std::upper_bound(mLevelBounds.cbegin(), mLevelBounds.cend(), iNodeIndex);
  return (mLevelBounds.cend() == wIt) ? mLevelBounds.back() : *wIt;
}

template <typename ArrayType>
std::vector<size_t> Flatbush<ArrayType>::search(const Box<ArrayType>& iBounds,
                                                const FilterCb& iFilterFn) const noexcept {
  const auto wCanLoop = canDoSearch(iBounds);
  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();
  auto wNodeIndex = mBoxes.size() - 1UL;
  std::queue<size_t> wQueue;
  std::vector<size_t> wResults;

  while (wCanLoop) {
    // find the end index of the node
    const size_t wEnd = std::min(wNodeIndex + wNodeSize, upperBound(wNodeIndex));

    // search through child nodes
    for (size_t wPosition = wNodeIndex; wPosition < wEnd; ++wPosition) {
      // check if node bbox intersects with query bbox
      if (iBounds.mMaxX < mBoxes[wPosition].mMinX /* maxX < nodeMinX */ ||
          iBounds.mMaxY < mBoxes[wPosition].mMinY /* maxY < nodeMinY */ ||
          iBounds.mMinX > mBoxes[wPosition].mMaxX /* minX > nodeMaxX */ ||
          iBounds.mMinY > mBoxes[wPosition].mMaxY /* minY > nodeMaxY */) {
        continue;
      }

      const size_t wIndex = mIsWideIndex ? mIndicesUint32[wPosition] : mIndicesUint16[wPosition];

      if (wNodeIndex >= wNumItems) {
        wQueue.push(wIndex);  // node; add it to the search queue
      } else if (!iFilterFn || iFilterFn(wIndex)) {
        wResults.push_back(wIndex);  // leaf item
      }
    }

    if (wQueue.empty()) {
      break;
    }

    wNodeIndex = wQueue.front() >> 2U;  // for binary compatibility with JS
    wQueue.pop();
  }

  return wResults;
}

template <typename ArrayType>
std::vector<size_t> Flatbush<ArrayType>::neighbors(const Point<ArrayType>& iPoint,
                                                   size_t iMaxResults,
                                                   double iMaxDistance,
                                                   const FilterCb& iFilterFn) const noexcept {
  const auto wCanLoop = canDoNeighbors(iPoint, iMaxResults, iMaxDistance);
  const auto wMaxDistSquared = iMaxDistance * iMaxDistance;
  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();
  auto wNodeIndex = mBoxes.size() - 1UL;
  std::priority_queue<IndexDistance> wQueue;
  std::vector<size_t> wResults;

  while (wCanLoop) {
    // find the end index of the node
    const auto wEnd = std::min(wNodeIndex + wNodeSize, upperBound(wNodeIndex));

    // search through child nodes
    for (auto wPosition = wNodeIndex; wPosition < wEnd; ++wPosition) {
      const size_t wIndex = (mIsWideIndex ? mIndicesUint32[wPosition] : mIndicesUint16[wPosition]);
      const auto wDistX = axisDistance(iPoint.mX, mBoxes[wPosition].mMinX, mBoxes[wPosition].mMaxX);
      const auto wDistY = axisDistance(iPoint.mY, mBoxes[wPosition].mMinY, mBoxes[wPosition].mMaxY);
      const auto wDistance = wDistX * wDistX + wDistY * wDistY;

      if (wDistance > wMaxDistSquared) {
        continue;
      } else if (wNodeIndex >= wNumItems) {
        wQueue.emplace(wIndex << 1U, wDistance);
      } else if (!iFilterFn || iFilterFn(wIndex)) {
        // put an odd index if it's an item rather than a node, to recognize later
        wQueue.emplace((wIndex << 1U) + 1U, wDistance);  // leaf node
      }
    }

    // pop items from the queue
    while (!wQueue.empty() && (wQueue.top().mId & 1U)) {
      wResults.push_back(wQueue.top().mId >> 1U);
      wQueue.pop();
      if (wResults.size() >= iMaxResults) {
        return wResults;
      }
    }

    if (wQueue.empty()) {
      break;
    }

    wNodeIndex = wQueue.top().mId >> 3U;  // 1 to undo indexing + 2 for binary compatibility with JS
    wQueue.pop();
  }

  return wResults;
}
}  // namespace flatbush

#endif  // FLATBUSH_H_
