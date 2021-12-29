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

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef MINIMAL_SPAN
#include <span>
#else
namespace std {
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
}
#endif // MINIMAL_SPAN

namespace flatbush {

  using FilterCb = std::function<bool(size_t)>;
  constexpr size_t gMaxUint32 = std::numeric_limits<size_t>::max();
  constexpr double gMaxDouble = std::numeric_limits<double>::max();
  constexpr uint8_t gInvalidArrayType = std::numeric_limits<uint8_t>::max();
  constexpr size_t gHeaderByteSize = 8;
  constexpr uint8_t gVersion = 3; // serialized format version

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

    uint32_t A, B, C, D;

    // Initial prefix scan round, prime with x and y
    {
      uint32_t a = x ^ y;
      uint32_t b = 0xFFFF ^ a;
      uint32_t c = 0xFFFF ^ (x | y);
      uint32_t d = x & (y ^ 0xFFFF);

      A = a | (b >> 1);
      B = (a >> 1) ^ a;

      C = ((c >> 1) ^ (b & (d >> 1))) ^ c;
      D = ((a & (c >> 1)) ^ (d >> 1)) ^ d;
    }

    {
      uint32_t a = A;
      uint32_t b = B;
      uint32_t c = C;
      uint32_t d = D;

      A = ((a & (a >> 2)) ^ (b & (b >> 2)));
      B = ((a & (b >> 2)) ^ (b & ((a ^ b) >> 2)));

      C ^= ((a & (c >> 2)) ^ (b & (d >> 2)));
      D ^= ((b & (c >> 2)) ^ ((a ^ b) & (d >> 2)));
    }

    {
      uint32_t a = A;
      uint32_t b = B;
      uint32_t c = C;
      uint32_t d = D;

      A = ((a & (a >> 4)) ^ (b & (b >> 4)));
      B = ((a & (b >> 4)) ^ (b & ((a ^ b) >> 4)));

      C ^= ((a & (c >> 4)) ^ (b & (d >> 4)));
      D ^= ((b & (c >> 4)) ^ ((a ^ b) & (d >> 4)));
    }

    // Final round and projection
    {
      uint32_t a = A;
      uint32_t b = B;
      uint32_t c = C;
      uint32_t d = D;

      C ^= ((a & (c >> 8)) ^ (b & (d >> 8)));
      D ^= ((b & (c >> 8)) ^ ((a ^ b) & (d >> 8)));
    }

    // Undo transformation prefix scan
    uint32_t a = C ^ (C >> 1);
    uint32_t b = D ^ (D >> 1);

    // Recover index bits
    uint32_t i0 = x ^ y;
    uint32_t i1 = b | (0xFFFF ^ (i0 | a));

    return ((Interleave(i1) << 1) | Interleave(i0)) >> (32 - 2 * n);
  }

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
    return gInvalidArrayType; // Should not happen
  }

  template <typename ArrayType>
  class Flatbush
  {
  public:
    ~Flatbush() = default;

    size_t add(ArrayType iMinX, ArrayType iMinY, ArrayType iMaxX, ArrayType iMaxY);
    void finish();
    std::vector<size_t> search(ArrayType iMinX, ArrayType iMinY, ArrayType iMaxX, ArrayType iMaxY, FilterCb iFilterFn = nullptr) const;
    std::vector<size_t> neighbors(ArrayType iX, ArrayType iY, size_t iMaxResults = gMaxUint32, double iMaxDistance = gMaxDouble, FilterCb iFilterFn = nullptr) const;
    uint16_t nodeSize() const;
    uint32_t numItems() const;
    size_t boxSize() const { return mBoxes.size(); }
    size_t indexSize() const { return mIsWideIndex ? mIndicesU32.size() : mIndicesU16.size(); }
    std::span<uint8_t const> data() const;

    static Flatbush<ArrayType> create(uint32_t iNumItems, uint16_t iNodeSize = 16);
    static Flatbush<ArrayType> from(const uint8_t* iData);

  private:
    explicit Flatbush(uint32_t iNumItems, uint16_t iNodeSize);
    explicit Flatbush(const uint8_t* iData);

    void init(uint32_t iNumItems, uint16_t iNodeSize);
    void sort(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight);
    void swap(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight);
    double axisDist(ArrayType iValue, ArrayType iMin, ArrayType iMax) const; 
    size_t upperBound(size_t iNodeIndex) const;

    struct IndexDistance
    {
      size_t mId;
      ArrayType mValue;
      bool operator< (const IndexDistance& iOther) const { return iOther.mValue < mValue; }
      bool operator> (const IndexDistance& iOther) const { return iOther.mValue > mValue; }
    };

    // views
    std::vector<uint8_t> mData;
    std::span<ArrayType> mBoxes;
    std::span<uint16_t> mIndicesU16;
    std::span<uint32_t> mIndicesU32;
    // pick appropriate index view
    bool mIsWideIndex;
    // box stuff
    size_t mPosition;
    std::vector<size_t> mLevelBounds;
    ArrayType mMinX;
    ArrayType mMinY;
    ArrayType mMaxX;
    ArrayType mMaxY;
  };

  template <typename ArrayType>
  Flatbush<ArrayType> Flatbush<ArrayType>::create(uint32_t iNumItems, uint16_t iNodeSize)
  {
    if (arrayTypeIndex<ArrayType>() == gInvalidArrayType)
    {
      throw std::runtime_error("Unexpected typed array class. Only integral and floating point allowed.");
    }

    if (iNumItems <= 0)
    {
      throw std::invalid_argument("Unpexpected numItems value: " + std::to_string(iNumItems) + ".");
    }

    return Flatbush<ArrayType>(iNumItems, iNodeSize);
  }

  template <typename ArrayType>
  Flatbush<ArrayType> Flatbush<ArrayType>::from(const uint8_t* iData)
  {
    if (!iData)
    {
      throw std::invalid_argument("Data is incomplete or missing.");
    }

    auto wMagic = iData[0];
    if (wMagic != 0xfb)
    {
      throw std::invalid_argument("Data does not appear to be in a Flatbush format.");
    }

    auto wVersionAndType = iData[1];
    if ((wVersionAndType >> 4) != gVersion)
    {
      throw std::invalid_argument("Got v" + std::to_string(wVersionAndType >> 4) + " data when expected v" + std::to_string(gVersion) + ".");
    }

    auto wArrayTypeIndex = arrayTypeIndex<ArrayType>();
    if (wArrayTypeIndex != (wVersionAndType & 0x0f))
    {
      throw std::runtime_error("Template type does not match encoded type.");
    }

    return Flatbush<ArrayType>(iData);
  }

  template <typename ArrayType>
  Flatbush<ArrayType>::Flatbush(uint32_t iNumItems, uint16_t iNodeSize)
  {
    iNodeSize = std::min(std::max(iNodeSize, static_cast<uint16_t>(2)), static_cast<uint16_t>(65535));
    init(iNumItems, iNodeSize);

    mData[0] = 0xfb;
    mData[1] = (gVersion << 4) + arrayTypeIndex<ArrayType>();
    *reinterpret_cast<uint16_t*>(&mData[2]) = iNodeSize;
    *reinterpret_cast<uint32_t*>(&mData[4]) = iNumItems;
    mPosition = 0;
    mMinX = std::numeric_limits<ArrayType>::max();
    mMinY = std::numeric_limits<ArrayType>::max();
    mMaxX = std::numeric_limits<ArrayType>::lowest();
    mMaxY = std::numeric_limits<ArrayType>::lowest();
  }

  template <typename ArrayType>
  Flatbush<ArrayType>::Flatbush(const uint8_t* iData)
  {
    uint16_t wNodeSize = *reinterpret_cast<const uint16_t*>(&iData[2]);
    uint32_t wNumItems = *reinterpret_cast<const uint32_t*>(&iData[4]);
    init(wNumItems, wNodeSize);

    memcpy(mData.data(), iData, mData.capacity());
    mPosition = mLevelBounds.back();
    mMinX = mBoxes[mPosition - 4];
    mMinY = mBoxes[mPosition - 3];
    mMaxX = mBoxes[mPosition - 2];
    mMaxY = mBoxes[mPosition - 1];
  }

  template <typename ArrayType>
  void Flatbush<ArrayType>::init(uint32_t iNumItems, uint16_t iNodeSize)
  {
    // calculate the total number of nodes in the R-tree to allocate space for
    // and the index of each tree level (used in search later)
    size_t wCount = iNumItems;
    size_t wNumNodes = iNumItems;
    mLevelBounds.push_back(wNumNodes * 4);

    do
    {
      wCount = static_cast<size_t>(std::ceil(static_cast<float>(wCount) / iNodeSize));
      wNumNodes += wCount;
      mLevelBounds.push_back(wNumNodes * 4);
    } 
    while (wCount > 1);

    // Sizes
    mIsWideIndex = wNumNodes >= 16384;
    size_t wIndexArrayTypeSize = mIsWideIndex ? sizeof(uint32_t) : sizeof(uint16_t);
    size_t wNodesByteSize = wNumNodes * 4 * sizeof(ArrayType);
    size_t wDataSize = gHeaderByteSize + wNodesByteSize + wNumNodes * wIndexArrayTypeSize;
    // Views
    mData.reserve(wDataSize);
    mBoxes = std::span<ArrayType>(reinterpret_cast<ArrayType*>(&mData[gHeaderByteSize]), wNumNodes * 4);
    mIndicesU16 = std::span<uint16_t>(reinterpret_cast<uint16_t*>(&mData[gHeaderByteSize + wNodesByteSize]), wNumNodes);
    mIndicesU32 = std::span<uint32_t>(reinterpret_cast<uint32_t*>(&mData[gHeaderByteSize + wNodesByteSize]), wNumNodes);
  }

  template <typename ArrayType>
  uint16_t Flatbush<ArrayType>::nodeSize() const
  {
    return mData[2] | mData[3] << 8;
  }

  template <typename ArrayType>
  uint32_t Flatbush<ArrayType>::numItems() const
  {
    return mData[4] | mData[5] << 8 | mData[6] << 16 | mData[7] << 24;
  }

  template <typename ArrayType>
  std::span<uint8_t const> Flatbush<ArrayType>::data() const
  {
    return { mData.data(), mData.capacity() };
  }

  template <typename ArrayType>
  size_t Flatbush<ArrayType>::add(ArrayType iMinX, ArrayType iMinY, ArrayType iMaxX, ArrayType iMaxY)
  {
    auto wIndex = mPosition >> 2;

    if (mIsWideIndex) mIndicesU32[wIndex] = static_cast<uint32_t>(wIndex);
    else mIndicesU16[wIndex] = static_cast<uint16_t>(wIndex);

    mBoxes[mPosition++] = iMinX;
    mBoxes[mPosition++] = iMinY;
    mBoxes[mPosition++] = iMaxX;
    mBoxes[mPosition++] = iMaxY;

    if (iMinX < mMinX) mMinX = iMinX;
    if (iMinY < mMinY) mMinY = iMinY;
    if (iMaxX > mMaxX) mMaxX = iMaxX;
    if (iMaxY > mMaxY) mMaxY = iMaxY;

    return wIndex;
  }

  template <typename ArrayType>
  void Flatbush<ArrayType>::finish() {
    auto wNumItems = numItems();

    if (mPosition >> 2 != wNumItems)
    {
      throw std::runtime_error("Added " + std::to_string(mPosition >> 2) + " items when expected " + std::to_string(wNumItems) + ".");
    }

    auto wNodeSize = nodeSize();

    if (wNumItems <= wNodeSize)
    {
      mBoxes[mPosition]     = mMinX;
      mBoxes[mPosition + 1] = mMinY;
      mBoxes[mPosition + 2] = mMaxX;
      mBoxes[mPosition + 3] = mMaxY;
      return;
    }

    ArrayType wWidth = mMaxX - mMinX;
    ArrayType wHeight = mMaxY - mMinY;
    std::vector<uint32_t> wHilbertValues(wNumItems);
    size_t wHilbertMax = (1 << 16) - 1;

    // map item centers into Hilbert coordinate space and calculate Hilbert values
    for (size_t wIdx = 0; wIdx < wNumItems; ++wIdx)
    {
      auto wPosition = wIdx << 2;
      auto wMinX = static_cast<double>(mBoxes[wPosition]);
      auto wMinY = static_cast<double>(mBoxes[wPosition + 1]);
      auto wMaxX = static_cast<double>(mBoxes[wPosition + 2]);
      auto wMaxY = static_cast<double>(mBoxes[wPosition + 3]);
      auto wX = static_cast<uint32_t>(std::floor((wMinX + wMaxX) / 2 - mMinX) / wWidth * wHilbertMax);
      auto wY = static_cast<uint32_t>(std::floor((wMinY + wMaxY) / 2 - mMinY) / wHeight * wHilbertMax);
      wHilbertValues[wIdx] = HilbertXYToIndex(16, wX, wY);
    }

    // sort items by their Hilbert value (for packing later)
    sort(wHilbertValues, 0, wNumItems - 1);

    for (size_t wIdx = 0, wPosition = 0; wIdx < mLevelBounds.size() - 1; ++wIdx)
    {
      auto wEnd = mLevelBounds[wIdx];

      // generate a parent node for each block of consecutive <nodeSize> nodes
      while (wPosition < wEnd)
      {
        auto wNodeIndex = wPosition;

        // calculate bbox for the new node
        auto wNodeMinX = mBoxes[wPosition];
        auto wNodeMinY = mBoxes[wPosition + 1];
        auto wNodeMaxX = mBoxes[wPosition + 2];
        auto wNodeMaxY = mBoxes[wPosition + 3];

        for (size_t wCount = 0; wCount < wNodeSize && wPosition < wEnd; ++wCount, wPosition += 4)
        {
          if (mBoxes[wPosition]     < wNodeMinX) wNodeMinX = mBoxes[wPosition];
          if (mBoxes[wPosition + 1] < wNodeMinY) wNodeMinY = mBoxes[wPosition + 1];
          if (mBoxes[wPosition + 2] > wNodeMaxX) wNodeMaxX = mBoxes[wPosition + 2];
          if (mBoxes[wPosition + 3] > wNodeMaxY) wNodeMaxY = mBoxes[wPosition + 3];
        }

        // add the new node to the tree data
        if (mIsWideIndex) mIndicesU32[(mPosition >> 2)] = static_cast<uint32_t>(wNodeIndex);
        else mIndicesU16[(mPosition >> 2)] = static_cast<uint16_t>(wNodeIndex);

        mBoxes[mPosition++] = wNodeMinX;
        mBoxes[mPosition++] = wNodeMinY;
        mBoxes[mPosition++] = wNodeMaxX;
        mBoxes[mPosition++] = wNodeMaxY;
      }
    }
  }

  // custom quicksort that partially sorts bbox data alongside the hilbert values
  template <typename ArrayType>
  void Flatbush<ArrayType>::sort(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight)
  {
    if (std::floor(iLeft / nodeSize()) < std::floor(iRight / nodeSize()))
    {
      auto wPivot = iValues[(iLeft + iRight) >> 1];
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
  void Flatbush<ArrayType>::swap(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight)
  {
    std::swap(iValues[iLeft], iValues[iRight]);

    auto wIdxLeft = iLeft << 2;
    auto wIdxRight = iRight << 2;

    std::swap(mBoxes[wIdxLeft], mBoxes[wIdxRight]);
    std::swap(mBoxes[wIdxLeft + 1], mBoxes[wIdxRight + 1]);
    std::swap(mBoxes[wIdxLeft + 2], mBoxes[wIdxRight + 2]);
    std::swap(mBoxes[wIdxLeft + 3], mBoxes[wIdxRight + 3]);

    if (mIsWideIndex) std::swap(mIndicesU32[iLeft], mIndicesU32[iRight]);
    else std::swap(mIndicesU16[iLeft], mIndicesU16[iRight]);
  }

  template <typename ArrayType>
  double Flatbush<ArrayType>::axisDist(ArrayType iValue, ArrayType iMin, ArrayType iMax) const
  {
    return iValue < iMin ? iMin - iValue : iValue <= iMax ? 0 : iValue - iMax;
  };

  template <typename ArrayType>
  size_t Flatbush<ArrayType>::upperBound(size_t iNodeIndex) const
  {
    auto wIt = std::upper_bound(mLevelBounds.begin(), mLevelBounds.end(), iNodeIndex);
    return (mLevelBounds.end() == wIt) ? mLevelBounds.back() : *wIt;
  }

  template <typename ArrayType>
  std::vector<size_t> Flatbush<ArrayType>::search(ArrayType iMinX, ArrayType iMinY, ArrayType iMaxX, ArrayType iMaxY, FilterCb iFilterFn) const
  {
    if (mPosition != mBoxes.size())
    {
      throw std::runtime_error("Data not yet indexed - call index.finish().");
    }

    size_t wNumItems = static_cast<size_t>(numItems()) << 2;
    size_t wNodeSize = static_cast<size_t>(nodeSize()) << 2;
    size_t wNodeIndex = mBoxes.size() - 4;
    std::vector<size_t> wQueue;
    std::vector<size_t> wResults;
    bool wHasWork = true;

    while (wHasWork)
    {
      // find the end index of the node
      size_t wEnd = std::min(wNodeIndex + wNodeSize, upperBound(wNodeIndex));

      // search through child nodes
      for (size_t wPosition = wNodeIndex; wPosition < wEnd; wPosition += 4)
      {
        size_t wIndex = (mIsWideIndex ? mIndicesU32[wPosition >> 2] : mIndicesU16[wPosition >> 2]) | 0;

        // check if node bbox intersects with query bbox
        if (iMaxX < mBoxes[wPosition]) continue; // maxX < nodeMinX
        if (iMaxY < mBoxes[wPosition + 1]) continue; // maxY < nodeMinY
        if (iMinX > mBoxes[wPosition + 2]) continue; // minX > nodeMaxX
        if (iMinY > mBoxes[wPosition + 3]) continue; // minY > nodeMaxY

        if (wNodeIndex < wNumItems)
        {
          if (!iFilterFn || iFilterFn(wIndex))
          {
            wResults.push_back(wIndex); // leaf item
          }
        }
        else
        {
          wQueue.push_back(wIndex); // node; add it to the search queue
        }
      }

      if (!wQueue.empty())
      {
        wNodeIndex = wQueue.back();
        wQueue.pop_back();
      }
      else
      {
        wHasWork = false;
      }
    }

    return wResults;
  }

  template <typename ArrayType>
  std::vector<size_t> Flatbush<ArrayType>::neighbors(ArrayType iX, ArrayType iY, size_t iMaxResults, double iMaxDistance, FilterCb iFilterFn) const
  {
    if (mPosition != mBoxes.size())
    {
      throw std::runtime_error("Data not yet indexed - call index.finish().");
    }

    std::priority_queue<IndexDistance> wQueue;
    std::vector<size_t> wResults;
    auto wNumItems = static_cast<size_t>(numItems()) << 2;
    auto wNodeSize = static_cast<size_t>(nodeSize()) << 2;
    auto wNodeIndex = mBoxes.size() - 4;
    auto wMaxDistSquared = iMaxDistance * iMaxDistance;
    bool wHasWork = true;

    while (wHasWork)
    {
      // find the end index of the node
      size_t wEnd = std::min(wNodeIndex + wNodeSize, upperBound(wNodeIndex));

      // search through child nodes
      for (size_t wPosition = wNodeIndex; wPosition < wEnd; wPosition += 4)
      {
        size_t wIndex = (mIsWideIndex ? mIndicesU32[wPosition >> 2] : mIndicesU16[wPosition >> 2]) | 0;

        auto wDistX = axisDist(iX, mBoxes[wPosition], mBoxes[wPosition + 2]);
        auto wDistY = axisDist(iY, mBoxes[wPosition + 1], mBoxes[wPosition + 3]);
        auto wDistance = wDistX * wDistX + wDistY * wDistY;

        if (wNodeIndex < wNumItems)
        { // leaf node
          if (!iFilterFn || iFilterFn(wIndex))
          {
            // put an odd index if it's an item rather than a node, to recognize later
            wQueue.push({ (wIndex << 1) + 1, wDistance });
          }
        }
        else
        {
          wQueue.push({ wIndex << 1, wDistance });
        }
      }

      // pop items from the queue
      while (wHasWork && !wQueue.empty() && (wQueue.top().mId & 1))
      {
        if (wQueue.top().mValue > wMaxDistSquared)
        {
          wHasWork = false;
        }
        else
        {
          wResults.push_back(wQueue.top().mId >> 1);
          wQueue.pop();

          if (wResults.size() == iMaxResults)
          {
            wHasWork = false;
          }
        }
      }

      if (wHasWork && !wQueue.empty())
      {
        wNodeIndex = wQueue.top().mId >> 1;
        wQueue.pop();
      }
      else
      {
        wHasWork = false;
      }
    }

    return wResults;
  }
}
