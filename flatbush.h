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

#ifndef LIB_FLATBUSH_H
#define LIB_FLATBUSH_H

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef FLATBUSH_SPAN
#include <span>
namespace flatbush {
  using std::span;
}
#else
namespace flatbush {

template<typename Type>
class span
{
  Type* mPtr;
  size_t mLen;

 public:
  span() noexcept
    : mPtr{ nullptr }
    , mLen{ 0 }
  {}

  span(Type* iPtr, size_t iLen) noexcept
    : mPtr{ iPtr }
    , mLen{ iLen }
  {}

  Type& operator[](size_t iIndex) noexcept
  {
    return mPtr[iIndex];
  }

  Type const& operator[](size_t iIndex) const noexcept
  {
    return mPtr[iIndex];
  }

  const Type* data() const noexcept
  {
    return mPtr;
  }

  size_t size() const noexcept
  {
    return mLen;
  }

  Type* begin() noexcept
  {
    return mPtr;
  }

  Type* end() noexcept
  {
    return mPtr + mLen;
  }
};

}  // namespace flatbush
#endif  // FLATBUSH_SPAN

namespace flatbush {

using FilterCb = std::function<bool(size_t)>;
constexpr auto gHilbertMax = std::numeric_limits<uint16_t>::max();
constexpr double gMaxDouble = std::numeric_limits<double>::max();
constexpr size_t gMaxUint32 = std::numeric_limits<size_t>::max();
constexpr size_t gDefaultNodeSize = 16;
constexpr size_t gHeaderByteSize = 8;
constexpr uint8_t gInvalidArrayType = std::numeric_limits<uint8_t>::max();
constexpr uint8_t gValidityFlag = 0xfb;
constexpr uint8_t gVersion = 3;  // serialized format version

namespace detail {

template<class To, class From>
To bit_cast(From const& from)
{
    static_assert(sizeof(To) == sizeof(From));

    To to;
    std::memcpy(&to, &from, sizeof(To));
    return to;
}

// From https://github.com/rawrunprotected/hilbert_curves (public domain)
inline uint32_t Interleave(uint32_t x)
{
  x = (x | (x << 8)) & 0x00FF00FF;
  x = (x | (x << 4)) & 0x0F0F0F0F;
  x = (x | (x << 2)) & 0x33333333;
  x = (x | (x << 1)) & 0x55555555;
  return x;
}

inline uint32_t HilbertXYToIndex(uint32_t n, uint32_t x, uint32_t y)
{
  x = x << (16 - n);
  y = y << (16 - n);

  // Initial prefix scan round, prime with x and y
  uint32_t a = x ^ y;
  uint32_t b = 0xFFFF ^ a;
  uint32_t c = 0xFFFF ^ (x | y);
  uint32_t d = x & (y ^ 0xFFFF);
  uint32_t A = a | (b >> 1);
  uint32_t B = (a >> 1) ^ a;
  uint32_t C = ((c >> 1) ^ (b & (d >> 1))) ^ c;
  uint32_t D = ((a & (c >> 1)) ^ (d >> 1)) ^ d;

  a = A;
  b = B;
  c = C;
  d = D;
  A = ((a & (a >> 2)) ^ (b & (b >> 2)));
  B = ((a & (b >> 2)) ^ (b & ((a ^ b) >> 2)));
  C ^= ((a & (c >> 2)) ^ (b & (d >> 2)));
  D ^= ((b & (c >> 2)) ^ ((a ^ b) & (d >> 2)));

  a = A;
  b = B;
  c = C;
  d = D;
  A = ((a & (a >> 4)) ^ (b & (b >> 4)));
  B = ((a & (b >> 4)) ^ (b & ((a ^ b) >> 4)));
  C ^= ((a & (c >> 4)) ^ (b & (d >> 4)));
  D ^= ((b & (c >> 4)) ^ ((a ^ b) & (d >> 4)));

  // Final round and projection
  a = A;
  b = B;
  c = C;
  d = D;
  C ^= ((a & (c >> 8)) ^ (b & (d >> 8)));
  D ^= ((b & (c >> 8)) ^ ((a ^ b) & (d >> 8)));

  // Undo transformation prefix scan
  a = C ^ (C >> 1);
  b = D ^ (D >> 1);

  // Recover index bits
  uint32_t i0 = x ^ y;
  uint32_t i1 = b | (0xFFFF ^ (i0 | a));

  return ((Interleave(i1) << 1) | Interleave(i0)) >> (32 - 2 * n);
}
}  // namespace detail

template <typename ArrayType>
inline uint8_t arrayTypeIndex()
{
  if (std::is_same<ArrayType, std::int8_t>::value)   return 0;
  if (std::is_same<ArrayType, std::uint8_t>::value)  return 1;
  if (std::is_same<ArrayType, std::int16_t>::value)  return 3;
  if (std::is_same<ArrayType, std::uint16_t>::value) return 4;
  if (std::is_same<ArrayType, std::int32_t>::value)  return 5;
  if (std::is_same<ArrayType, std::uint32_t>::value) return 6;
  if (std::is_same<ArrayType, float>::value)         return 7;
  if (std::is_same<ArrayType, double>::value)        return 8;
  return gInvalidArrayType;
}

inline std::string arrayTypeName(size_t iIndex)
{
  static const std::array<std::string, 9> sArrayTypeNames
  {
    "int8_t", "uint8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t", "float", "double"
  };

  return iIndex < sArrayTypeNames.size() ? sArrayTypeNames[iIndex] : "unknown";
}

template <typename ArrayType>
struct Box
{
  ArrayType mMinX;
  ArrayType mMinY;
  ArrayType mMaxX;
  ArrayType mMaxY;
};

template <typename ArrayType>
struct Point
{
  ArrayType mX;
  ArrayType mY;
};

template <typename ArrayType> class Flatbush;
template<class ArrayType>
class FlatbushBuilder
{
 public:
  explicit FlatbushBuilder(size_t iNumItems = 10, uint16_t iNodeSize = gDefaultNodeSize)
    : mNodeSize(iNodeSize)
  {
    if (arrayTypeIndex<ArrayType>() == gInvalidArrayType)
    {
      throw std::invalid_argument("Unexpected typed array class. Expecting non 64-bit integral or floating point.");
    }
    mItems.reserve(iNumItems);
  };

  inline void clear() noexcept
  {
    mItems.clear();
  };

  inline size_t add(const Box<ArrayType>& iBox) noexcept
  {
    mItems.push_back(iBox);
    return mItems.size() - 1;
  };

  Flatbush<ArrayType> finish() const;
  static Flatbush<ArrayType> from(const uint8_t* iData);

 private:
  std::uint16_t mNodeSize;
  std::vector<Box<ArrayType>> mItems;
};

template <typename ArrayType>
Flatbush<ArrayType> FlatbushBuilder<ArrayType>::finish() const
{
  if (mItems.empty())
  {
    throw std::invalid_argument("No items have been added. Nothing to build.");
  }

  Flatbush<ArrayType> wIndex(uint32_t(mItems.size()), mNodeSize);
  for (const auto& wItem : mItems)
  {
    wIndex.add(wItem);
  }
  wIndex.finish();

  return wIndex;
};

template <typename ArrayType>
Flatbush<ArrayType> FlatbushBuilder<ArrayType>::from(const uint8_t* iData)
{
  if (!iData)
  {
    throw std::invalid_argument("Data is incomplete or missing.");
  }

  auto wMagic = iData[0];
  if (wMagic != gValidityFlag)
  {
    throw std::invalid_argument("Data does not appear to be in a Flatbush format.");
  }

  auto wEncodedVersion = iData[1] >> 4;
  if (wEncodedVersion != gVersion)
  {
    throw std::invalid_argument("Got v" + std::to_string(wEncodedVersion) + " data when expected v" + std::to_string(gVersion) + ".");
  }

  auto wExpectedType = arrayTypeIndex<ArrayType>();
  auto wEncodedType = iData[1] & 0x0f;
  if (wExpectedType != wEncodedType)
  {
    throw std::invalid_argument("Expected type is " + arrayTypeName(wEncodedType) + ", but got template type " + arrayTypeName(wExpectedType));
  }

  return Flatbush<ArrayType>(iData);
}

template <typename ArrayType>
class Flatbush
{
 public:
  Flatbush(const Flatbush&) = delete;
  Flatbush& operator=(const Flatbush&) = delete;
  Flatbush(Flatbush&&) noexcept = default;
  Flatbush& operator=(Flatbush&&) noexcept = default;
  ~Flatbush() = default;

  std::vector<size_t> search(Box<ArrayType> iBounds, const FilterCb& iFilterFn = nullptr) const noexcept;
  std::vector<size_t> neighbors(Point<ArrayType> iPoint, size_t iMaxResults = gMaxUint32, double iMaxDistance = gMaxDouble, const FilterCb& iFilterFn = nullptr) const noexcept;
  inline size_t nodeSize() const noexcept { return mData[2] | mData[3] << 8; };
  inline size_t numItems() const noexcept { return mData[4] | mData[5] << 8 | mData[6] << 16 | mData[7] << 24; };
  inline size_t boxSize() const noexcept { return mBoxes.size() * 4; };
  inline size_t indexSize() const noexcept { return mIsWideIndex ? mIndicesUint32.size() : mIndicesUint16.size(); };
  inline span<const uint8_t> data() const noexcept { return { mData.data(), mData.capacity() }; };

  friend class FlatbushBuilder<ArrayType>;

 private:
  static constexpr ArrayType cMaxValue = std::numeric_limits<ArrayType>::max();
  static constexpr ArrayType cMinValue = std::numeric_limits<ArrayType>::lowest();
  static inline double axisDistance(ArrayType iValue, ArrayType iMin, ArrayType iMax) noexcept
  {
    return iValue < iMin ? iMin - iValue : std::max(iValue - iMax, 0.0);
  }

  explicit Flatbush(uint32_t iNumItems, uint16_t iNodeSize) noexcept;
  explicit Flatbush(const uint8_t* iData) noexcept;

  size_t add(const Box<ArrayType>& iBox) noexcept;
  void finish() noexcept;
  void init(uint32_t iNumItems, uint16_t iNodeSize) noexcept;
  void sort(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight) noexcept;
  void swap(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight) noexcept;
  size_t upperBound(size_t iNodeIndex) const noexcept;

  struct IndexDistance
  {
    size_t mId;
    ArrayType mDistance;
    bool operator< (const IndexDistance& iOther) const { return iOther.mDistance < mDistance; }
    bool operator> (const IndexDistance& iOther) const { return iOther.mDistance > mDistance; }
  };

  // views
  std::vector<uint8_t> mData;
  span<Box<ArrayType>> mBoxes;
  span<uint16_t> mIndicesUint16;
  span<uint32_t> mIndicesUint32;
  // pick appropriate index view
  bool mIsWideIndex;
  // box stuff
  size_t mPosition = 0;
  std::vector<size_t> mLevelBounds;
  Box<ArrayType> mBounds;
};

template <typename ArrayType>
Flatbush<ArrayType>::Flatbush(uint32_t iNumItems, uint16_t iNodeSize) noexcept
{
  iNodeSize = std::min(std::max(iNodeSize, uint16_t{ 2 }), uint16_t{ 65535 });
  init(iNumItems, iNodeSize);

  mData.assign(mData.capacity(), 0);
  mData[0] = gValidityFlag;
  mData[1] = (gVersion << 4) + arrayTypeIndex<ArrayType>();
  *detail::bit_cast<uint16_t*>(&mData[2]) = iNodeSize;
  *detail::bit_cast<uint32_t*>(&mData[4]) = iNumItems;
  mBounds = { cMaxValue, cMaxValue, cMinValue, cMinValue };
}

template <typename ArrayType>
Flatbush<ArrayType>::Flatbush(const uint8_t* iData) noexcept
{
  auto wNodeSize = *detail::bit_cast<const uint16_t*>(&iData[2]);
  auto wNumItems = *detail::bit_cast<const uint32_t*>(&iData[4]);
  init(wNumItems, wNodeSize);

  mData.insert(mData.begin(), &iData[0], &iData[mData.capacity()]);
  mPosition = mLevelBounds.back();
  mBounds = mBoxes[mPosition - 1];
}

template <typename ArrayType>
void Flatbush<ArrayType>::init(uint32_t iNumItems, uint16_t iNodeSize) noexcept
{
  // calculate the total number of nodes in the R-tree to allocate space for
  // and the index of each tree level (used in search later)
  size_t wCount = iNumItems;
  size_t wNumNodes = iNumItems;
  mLevelBounds.push_back(wNumNodes);

  do
  {
    wCount = (wCount + iNodeSize - 1) / iNodeSize;
    wNumNodes += wCount;
    mLevelBounds.push_back(wNumNodes);
  } while (wCount > 1);

  // Sizes
  mIsWideIndex = wNumNodes >= 16384;
  const size_t wIndicesByteSize = wNumNodes * (mIsWideIndex ? sizeof(uint32_t) : sizeof(uint16_t));
  const size_t wNodesByteSize = wNumNodes * sizeof(Box<ArrayType>);
  const size_t wDataSize = gHeaderByteSize + wNodesByteSize + wIndicesByteSize;
  // Views
  mData.reserve(wDataSize);
  mBoxes = span<Box<ArrayType>>(detail::bit_cast<Box<ArrayType>*>(&mData[gHeaderByteSize]), wNumNodes);
  mIndicesUint16 = span<uint16_t>(detail::bit_cast<uint16_t*>(&mData[gHeaderByteSize + wNodesByteSize]), wNumNodes);
  mIndicesUint32 = span<uint32_t>(detail::bit_cast<uint32_t*>(&mData[gHeaderByteSize + wNodesByteSize]), wNumNodes);
}

template <typename ArrayType>
size_t Flatbush<ArrayType>::add(const Box<ArrayType>& iBox) noexcept
{
  if (mIsWideIndex) mIndicesUint32[mPosition] = uint32_t(mPosition);
  else mIndicesUint16[mPosition] = uint16_t(mPosition);

  mBoxes[mPosition] = iBox;
  mBounds.mMinX = std::min(mBounds.mMinX, iBox.mMinX);
  mBounds.mMinY = std::min(mBounds.mMinY, iBox.mMinY);
  mBounds.mMaxX = std::max(mBounds.mMaxX, iBox.mMaxX);
  mBounds.mMaxY = std::max(mBounds.mMaxY, iBox.mMaxY);

  return mPosition++;
}

template <typename ArrayType>
void Flatbush<ArrayType>::finish() noexcept
{
  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();

  if (wNumItems <= wNodeSize)
  {
    mBoxes[mPosition++] = mBounds;
    return;
  }

  std::vector<uint32_t> wHilbertValues(wNumItems);
  const auto wHilbertWidth = gHilbertMax / (mBounds.mMaxX - mBounds.mMinX);
  const auto wHilbertHeight = gHilbertMax / (mBounds.mMaxY - mBounds.mMinY);

  // map item centers into Hilbert coordinate space and calculate Hilbert values
  for (size_t wIdx = 0; wIdx < wNumItems; ++wIdx)
  {
    wHilbertValues[wIdx] = detail::HilbertXYToIndex(
      16,
      uint32_t(wHilbertWidth * ((mBoxes[wIdx].mMinX + mBoxes[wIdx].mMaxX) / 2 - mBounds.mMinX)),
      uint32_t(wHilbertHeight * ((mBoxes[wIdx].mMinY + mBoxes[wIdx].mMaxY) / 2 - mBounds.mMinY))
    );
  }

  // sort items by their Hilbert value (for packing later)
  sort(wHilbertValues, 0, wNumItems - 1);

  for (size_t wIdx = 0, wPosition = 0; wIdx < mLevelBounds.size() - 1; ++wIdx)
  {
    const auto& wEnd = mLevelBounds[wIdx];

    // generate a parent node for each block of consecutive <nodeSize> nodes
    while (wPosition < wEnd)
    {
      const auto wNodeIndex = wPosition << 2;  // need to shift for binary compatibility with JS
      auto wNodeBox = mBoxes[wPosition];

      // calculate bbox for the new node
      for (size_t wCount = 0; wCount < wNodeSize && wPosition < wEnd; ++wCount, ++wPosition)
      {
        wNodeBox.mMinX = std::min(wNodeBox.mMinX, mBoxes[wPosition].mMinX);
        wNodeBox.mMinY = std::min(wNodeBox.mMinY, mBoxes[wPosition].mMinY);
        wNodeBox.mMaxX = std::max(wNodeBox.mMaxX, mBoxes[wPosition].mMaxX);
        wNodeBox.mMaxY = std::max(wNodeBox.mMaxY, mBoxes[wPosition].mMaxY);
      }

      // add the new node to the tree data
      if (mIsWideIndex) mIndicesUint32[mPosition] = uint32_t(wNodeIndex);
      else mIndicesUint16[mPosition] = uint16_t(wNodeIndex);

      mBoxes[mPosition++] = wNodeBox;
    }
  }
}

// custom quicksort that partially sorts bbox data alongside the hilbert values
template <typename ArrayType>
void Flatbush<ArrayType>::sort(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight) noexcept
{
  if (iLeft < iRight)
  {
    const auto wPivot = iValues[(iLeft + iRight) >> 1];
    auto wPivotLeft = iLeft - 1;
    auto wPivotRight = iRight + 1;

    while (true)
    {
      do wPivotLeft++; while (iValues[wPivotLeft] < wPivot);
      do wPivotRight--; while (iValues[wPivotRight] > wPivot);
      if (wPivotLeft >= wPivotRight) break;
      swap(iValues, wPivotLeft, wPivotRight);
    }

    sort(iValues, iLeft, wPivotRight);
    sort(iValues, wPivotRight + 1, iRight);
  }
}

// swap two values and two corresponding boxes
template <typename ArrayType>
void Flatbush<ArrayType>::swap(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight) noexcept
{
  std::swap(iValues[iLeft], iValues[iRight]);
  std::swap(mBoxes[iLeft], mBoxes[iRight]);

  if (mIsWideIndex) std::swap(mIndicesUint32[iLeft], mIndicesUint32[iRight]);
  else std::swap(mIndicesUint16[iLeft], mIndicesUint16[iRight]);
}

template <typename ArrayType>
size_t Flatbush<ArrayType>::upperBound(size_t iNodeIndex) const noexcept
{
  auto wIt = std::upper_bound(mLevelBounds.begin(), mLevelBounds.end(), iNodeIndex);
  return (mLevelBounds.end() == wIt) ? mLevelBounds.back() : *wIt;
}

template <typename ArrayType>
std::vector<size_t> Flatbush<ArrayType>::search(Box<ArrayType> iBounds, const FilterCb& iFilterFn) const noexcept
{
  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();
  auto wNodeIndex = mBoxes.size() - 1;
  std::vector<size_t> wQueue;
  std::vector<size_t> wResults;

  while (true)
  {
    // find the end index of the node
    const size_t wEnd = std::min(wNodeIndex + wNodeSize, upperBound(wNodeIndex));

    // search through child nodes
    for (size_t wPosition = wNodeIndex; wPosition < wEnd; ++wPosition)
    {
      // check if node bbox intersects with query bbox
      if (iBounds.mMaxX < mBoxes[wPosition].mMinX) continue;  // maxX < nodeMinX
      if (iBounds.mMaxY < mBoxes[wPosition].mMinY) continue;  // maxY < nodeMinY
      if (iBounds.mMinX > mBoxes[wPosition].mMaxX) continue;  // minX > nodeMaxX
      if (iBounds.mMinY > mBoxes[wPosition].mMaxY) continue;  // minY > nodeMaxY

      const auto wIndex = (mIsWideIndex ? mIndicesUint32[wPosition] : mIndicesUint16[wPosition]) | 0;

      if (wNodeIndex >= wNumItems)
      {
        wQueue.push_back(wIndex);  // node; add it to the search queue
      }
      else if (!iFilterFn || iFilterFn(wIndex))
      {
        wResults.push_back(wIndex);  // leaf item
      }
    }

    if (wQueue.empty()) break;
    wNodeIndex = wQueue.back() >> 2;  // need to shift for binary compatibility with JS
    wQueue.pop_back();
  }

  return wResults;
}

template <typename ArrayType>
std::vector<size_t> Flatbush<ArrayType>::neighbors(Point<ArrayType> iPoint, size_t iMaxResults, double iMaxDistance, const FilterCb& iFilterFn) const noexcept
{
  std::priority_queue<IndexDistance> wQueue;
  std::vector<size_t> wResults;
  const auto wMaxDistSquared = iMaxDistance * iMaxDistance;
  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();
  auto wNodeIndex = mBoxes.size() - 1;

  while (true)
  {
    // find the end index of the node
    const auto wEnd = std::min(wNodeIndex + wNodeSize, upperBound(wNodeIndex));

    // search through child nodes
    for (auto wPosition = wNodeIndex; wPosition < wEnd; ++wPosition)
    {
      const size_t wIndex = (mIsWideIndex ? mIndicesUint32[wPosition] : mIndicesUint16[wPosition]) | 0;
      const auto wDistX = axisDistance(iPoint.mX, mBoxes[wPosition].mMinX, mBoxes[wPosition].mMaxX);
      const auto wDistY = axisDistance(iPoint.mY, mBoxes[wPosition].mMinY, mBoxes[wPosition].mMaxY);
      const auto wDistance = wDistX * wDistX + wDistY * wDistY;
      if (wDistance > wMaxDistSquared) continue;

      if (wNodeIndex >= wNumItems)
      {
        wQueue.push({ wIndex << 1, wDistance });
      }
      else if (!iFilterFn || iFilterFn(wIndex))  // leaf node
      {
        // put an odd index if it's an item rather than a node, to recognize later
        wQueue.push({ (wIndex << 1) + 1, wDistance });
      }
    }

    // pop items from the queue
    while (!wQueue.empty() && (wQueue.top().mId & 1))
    {
      if (wQueue.top().mDistance > wMaxDistSquared) return wResults;
      wResults.push_back(wQueue.top().mId >> 1);
      wQueue.pop();
      if (wResults.size() >= iMaxResults) return wResults;
    }

    if (wQueue.empty()) break;
    wNodeIndex = wQueue.top().mId >> 3;  // 1 to undo queue indexing + 2 for binary compatibility with JS
    wQueue.pop();
  }

  return wResults;
}
}  // namespace flatbush

#endif  // LIB_FLATBUSH_H
