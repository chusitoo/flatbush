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
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
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

}
#endif // FLATBUSH_SPAN

namespace flatbush {

  using FilterCb = std::function<bool(size_t)>;
  constexpr auto gHilbertMax = std::numeric_limits<uint16_t>::max();
  constexpr double gMaxDouble = std::numeric_limits<double>::max();
  constexpr size_t gMaxUint32 = std::numeric_limits<size_t>::max();
  constexpr size_t gDefaultNodeSize = 16;
  constexpr size_t gHeaderByteSize = 8;
  constexpr uint8_t gInvalidArrayType = std::numeric_limits<uint8_t>::max();
  constexpr uint8_t gValidityFlag = 0xfb;
  constexpr uint8_t gVersion = 3; // serialized format version

  namespace detail {
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
    return gInvalidArrayType;
  }

  inline std::string arrayTypeName(size_t iIndex)
  {
    static const std::string sArrayTypeNames[9]
    {
      "int8_t", "uint8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t", "float", "double"
    };

    return iIndex < 9 ? sArrayTypeNames[iIndex] : "unknown";
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
    explicit FlatbushBuilder(uint32_t iNumItems = 10, uint16_t iNodeSize = gDefaultNodeSize)
      : mNodeSize(iNodeSize)
    {
      if (arrayTypeIndex<ArrayType>() == gInvalidArrayType)
      {
        throw std::runtime_error("Unexpected typed array class. Expecting non 64-bit integral or floating point.");
      }
      mItems.reserve(iNumItems);
    };

    inline void clear() noexcept
    {
      mItems.clear();
    };

    inline size_t add(Box<ArrayType> iBox) noexcept
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
      throw std::invalid_argument("Unpexpected numItems value: " + std::to_string(mItems.size()) + ".");
    }

    Flatbush<ArrayType> wIndex(static_cast<uint32_t>(mItems.size()), mNodeSize);
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
      throw std::runtime_error("Expected type is " + arrayTypeName(wEncodedType) + ", but got template type " + arrayTypeName(wExpectedType));
    }

    return Flatbush<ArrayType>(iData);
  }

  template <typename ArrayType>
  class Flatbush
  {
  public:
    Flatbush(const Flatbush&) = delete;
    Flatbush& operator=(const Flatbush&) = delete;
    Flatbush(Flatbush&&) = default;
    Flatbush& operator=(Flatbush&&) = default;
    ~Flatbush() = default;

    std::vector<size_t> search(Box<ArrayType> iBounds, FilterCb iFilterFn = nullptr) const noexcept;
    std::vector<size_t> neighbors(Point<ArrayType> iTarget, size_t iMaxResults = gMaxUint32, double iMaxDistance = gMaxDouble, FilterCb iFilterFn = nullptr) const noexcept;
    inline uint16_t nodeSize() const noexcept { return mData[2] | mData[3] << 8; };
    inline uint32_t numItems() const noexcept { return mData[4] | mData[5] << 8 | mData[6] << 16 | mData[7] << 24; };
    inline size_t boxSize() const noexcept { return mBoxes.size(); };
    inline size_t indexSize() const noexcept { return mIsWideIndex ? mIndicesUint32.size() : mIndicesUint16.size(); };
    inline span<const uint8_t> data() const noexcept { return { mData.data(), mData.capacity() }; };

    friend class FlatbushBuilder<ArrayType>;

  private:
    explicit Flatbush(uint32_t iNumItems, uint16_t iNodeSize) noexcept;
    explicit Flatbush(const uint8_t* iData) noexcept;

    size_t add(Box<ArrayType> iBox) noexcept;
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
    span<ArrayType> mBoxes;
    span<uint16_t> mIndicesUint16;
    span<uint32_t> mIndicesUint32;
    // pick appropriate index view
    bool mIsWideIndex;
    // box stuff
    size_t mPosition;
    std::vector<size_t> mLevelBounds;
    Box<ArrayType> mBounds;
  };

  template <typename ArrayType>
  Flatbush<ArrayType>::Flatbush(uint32_t iNumItems, uint16_t iNodeSize) noexcept
  {
    iNodeSize = std::min(std::max(iNodeSize, static_cast<uint16_t>(2)), static_cast<uint16_t>(65535));
    init(iNumItems, iNodeSize);

    mData.assign(mData.capacity(), 0);
    mData[0] = gValidityFlag;
    mData[1] = (gVersion << 4) + arrayTypeIndex<ArrayType>();
    *reinterpret_cast<uint16_t*>(&mData[2]) = iNodeSize;
    *reinterpret_cast<uint32_t*>(&mData[4]) = iNumItems;
    mPosition = 0;
    mBounds.mMinX = std::numeric_limits<ArrayType>::max();
    mBounds.mMinY = std::numeric_limits<ArrayType>::max();
    mBounds.mMaxX = std::numeric_limits<ArrayType>::lowest();
    mBounds.mMaxY = std::numeric_limits<ArrayType>::lowest();
  }

  template <typename ArrayType>
  Flatbush<ArrayType>::Flatbush(const uint8_t* iData) noexcept
  {
    auto wNodeSize = *reinterpret_cast<const uint16_t*>(&iData[2]);
    auto wNumItems = *reinterpret_cast<const uint32_t*>(&iData[4]);
    init(wNumItems, wNodeSize);

    mData.insert(mData.begin(), &iData[0], &iData[mData.capacity()]);
    mPosition = mLevelBounds.back();
    mBounds.mMinX = mBoxes[mPosition - 4];
    mBounds.mMinY = mBoxes[mPosition - 3];
    mBounds.mMaxX = mBoxes[mPosition - 2];
    mBounds.mMaxY = mBoxes[mPosition - 1];
  }

  template <typename ArrayType>
  void Flatbush<ArrayType>::init(uint32_t iNumItems, uint16_t iNodeSize) noexcept
  {
    // calculate the total number of nodes in the R-tree to allocate space for
    // and the index of each tree level (used in search later)
    size_t wCount = iNumItems;
    size_t wNumNodes = iNumItems;
    mLevelBounds.push_back(wNumNodes * 4);

    do
    {
      wCount = (wCount + iNodeSize - 1) / iNodeSize;
      wNumNodes += wCount;
      mLevelBounds.push_back(wNumNodes * 4);
    } while (wCount > 1);

    // Sizes
    mIsWideIndex = wNumNodes >= 16384;
    const size_t wIndexArrayTypeSize = mIsWideIndex ? sizeof(uint32_t) : sizeof(uint16_t);
    const size_t wNodesByteSize = wNumNodes * 4 * sizeof(ArrayType);
    const size_t wDataSize = gHeaderByteSize + wNodesByteSize + wNumNodes * wIndexArrayTypeSize;
    // Views
    mData.reserve(wDataSize);
    mBoxes = span<ArrayType>(reinterpret_cast<ArrayType*>(&mData[gHeaderByteSize]), wNumNodes * 4);
    mIndicesUint16 = span<uint16_t>(reinterpret_cast<uint16_t*>(&mData[gHeaderByteSize + wNodesByteSize]), wNumNodes);
    mIndicesUint32 = span<uint32_t>(reinterpret_cast<uint32_t*>(&mData[gHeaderByteSize + wNodesByteSize]), wNumNodes);
  }

  template <typename ArrayType>
  size_t Flatbush<ArrayType>::add(Box<ArrayType> iBox) noexcept
  {
    const auto wIndex = mPosition >> 2;

    if (mIsWideIndex) mIndicesUint32[wIndex] = static_cast<uint32_t>(wIndex);
    else mIndicesUint16[wIndex] = static_cast<uint16_t>(wIndex);

    mBoxes[mPosition++] = iBox.mMinX;
    mBoxes[mPosition++] = iBox.mMinY;
    mBoxes[mPosition++] = iBox.mMaxX;
    mBoxes[mPosition++] = iBox.mMaxY;

    if (iBox.mMinX < mBounds.mMinX) mBounds.mMinX = iBox.mMinX;
    if (iBox.mMinY < mBounds.mMinY) mBounds.mMinY = iBox.mMinY;
    if (iBox.mMaxX > mBounds.mMaxX) mBounds.mMaxX = iBox.mMaxX;
    if (iBox.mMaxY > mBounds.mMaxY) mBounds.mMaxY = iBox.mMaxY;

    return wIndex;
  }

  template <typename ArrayType>
  void Flatbush<ArrayType>::finish() noexcept
  {
    const auto wNumItems = numItems();
    const auto wNodeSize = nodeSize();

    if (wNumItems <= wNodeSize)
    {
      mBoxes[mPosition++] = mBounds.mMinX;
      mBoxes[mPosition++] = mBounds.mMinY;
      mBoxes[mPosition++] = mBounds.mMaxX;
      mBoxes[mPosition++] = mBounds.mMaxY;
      return;
    }

    std::vector<uint32_t> wHilbertValues(wNumItems);
    const auto wHilbertWidth = gHilbertMax / (mBounds.mMaxX - mBounds.mMinX);
    const auto wHilbertHeight = gHilbertMax / (mBounds.mMaxY - mBounds.mMinY);

    // map item centers into Hilbert coordinate space and calculate Hilbert values
    for (size_t wIdx = 0, wPosition = 0; wIdx < wNumItems; ++wIdx)
    {
      const auto& wMinX = mBoxes[wPosition++];
      const auto& wMinY = mBoxes[wPosition++];
      const auto& wMaxX = mBoxes[wPosition++];
      const auto& wMaxY = mBoxes[wPosition++];
      wHilbertValues[wIdx] = detail::HilbertXYToIndex(
        16,
        static_cast<uint32_t>(wHilbertWidth * ((wMinX + wMaxX) / 2 - mBounds.mMinX)),
        static_cast<uint32_t>(wHilbertHeight * ((wMinY + wMaxY) / 2 - mBounds.mMinY))
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
        const auto wNodeIndex = wPosition;

        // calculate bbox for the new node
        auto wNodeMinX = mBoxes[wPosition];
        auto wNodeMinY = mBoxes[wPosition + 1];
        auto wNodeMaxX = mBoxes[wPosition + 2];
        auto wNodeMaxY = mBoxes[wPosition + 3];

        for (size_t wCount = 0; wCount < wNodeSize && wPosition < wEnd; ++wCount, wPosition += 4)
        {
          if (mBoxes[wPosition] < wNodeMinX) wNodeMinX = mBoxes[wPosition];
          if (mBoxes[wPosition + 1] < wNodeMinY) wNodeMinY = mBoxes[wPosition + 1];
          if (mBoxes[wPosition + 2] > wNodeMaxX) wNodeMaxX = mBoxes[wPosition + 2];
          if (mBoxes[wPosition + 3] > wNodeMaxY) wNodeMaxY = mBoxes[wPosition + 3];
        }

        // add the new node to the tree data
        if (mIsWideIndex) mIndicesUint32[(mPosition >> 2)] = static_cast<uint32_t>(wNodeIndex);
        else mIndicesUint16[(mPosition >> 2)] = static_cast<uint16_t>(wNodeIndex);

        mBoxes[mPosition++] = wNodeMinX;
        mBoxes[mPosition++] = wNodeMinY;
        mBoxes[mPosition++] = wNodeMaxX;
        mBoxes[mPosition++] = wNodeMaxY;
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

    const auto wIdxLeft = iLeft << 2;
    const auto wIdxRight = iRight << 2;

    std::swap(mBoxes[wIdxLeft], mBoxes[wIdxRight]);
    std::swap(mBoxes[wIdxLeft + 1], mBoxes[wIdxRight + 1]);
    std::swap(mBoxes[wIdxLeft + 2], mBoxes[wIdxRight + 2]);
    std::swap(mBoxes[wIdxLeft + 3], mBoxes[wIdxRight + 3]);

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
  std::vector<size_t> Flatbush<ArrayType>::search(Box<ArrayType> iBounds, FilterCb iFilterFn) const noexcept
  {
    const auto wNumItems = numItems() << 2;
    const auto wNodeSize = nodeSize() << 2;
    auto wNodeIndex = mBoxes.size() - 4;
    std::vector<size_t> wQueue;
    std::vector<size_t> wResults;

    while (true)
    {
      // find the end index of the node
      const size_t wEnd = std::min(wNodeIndex + wNodeSize, upperBound(wNodeIndex));

      // search through child nodes
      for (size_t wPosition = wNodeIndex; wPosition < wEnd; wPosition += 4)
      {
        // check if node bbox intersects with query bbox
        if (iBounds.mMaxX < mBoxes[wPosition])     continue; // maxX < nodeMinX
        if (iBounds.mMaxY < mBoxes[wPosition + 1]) continue; // maxY < nodeMinY
        if (iBounds.mMinX > mBoxes[wPosition + 2]) continue; // minX > nodeMaxX
        if (iBounds.mMinY > mBoxes[wPosition + 3]) continue; // minY > nodeMaxY

        const auto wIndex = (mIsWideIndex ? mIndicesUint32[wPosition >> 2] : mIndicesUint16[wPosition >> 2]) | 0;

        if (wNodeIndex >= wNumItems)
        {
          wQueue.push_back(wIndex); // node; add it to the search queue
        }
        else if (!iFilterFn || iFilterFn(wIndex))
        {
          wResults.push_back(wIndex); // leaf item
        }
      }

      if (wQueue.empty()) break;
      wNodeIndex = wQueue.back();
      wQueue.pop_back();
    }

    return wResults;
  }

  template <typename ArrayType>
  std::vector<size_t> Flatbush<ArrayType>::neighbors(Point<ArrayType> iTarget, size_t iMaxResults, double iMaxDistance, FilterCb iFilterFn) const noexcept
  {
    auto wAxisDist = [](ArrayType iValue, ArrayType iMin, ArrayType iMax)
    {
      return iValue < iMin ? iMin - iValue : iValue <= iMax ? 0 : iValue - iMax;
    };

    std::priority_queue<IndexDistance> wQueue;
    std::vector<size_t> wResults;
    const auto wMaxDistSquared = iMaxDistance * iMaxDistance;
    const auto wNumItems = numItems() << 2;
    const auto wNodeSize = nodeSize() << 2;
    auto wNodeIndex = mBoxes.size() - 4;

    while (true)
    {
      // find the end index of the node
      const auto wEnd = std::min(wNodeIndex + wNodeSize, upperBound(wNodeIndex));

      // search through child nodes
      for (size_t wPosition = wNodeIndex; wPosition < wEnd; wPosition += 4)
      {
        const size_t wIndex = (mIsWideIndex ? mIndicesUint32[wPosition >> 2] : mIndicesUint16[wPosition >> 2]) | 0;
        const auto wDistX = wAxisDist(iTarget.mX, mBoxes[wPosition], mBoxes[wPosition + 2]);
        const auto wDistY = wAxisDist(iTarget.mY, mBoxes[wPosition + 1], mBoxes[wPosition + 3]);
        const auto wDistance = wDistX * wDistX + wDistY * wDistY;

        if (wNodeIndex >= wNumItems)
        {
          wQueue.push({ wIndex << 1, wDistance });
        }
        else if (!iFilterFn || iFilterFn(wIndex)) // leaf node
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
      wNodeIndex = wQueue.top().mId >> 1;
      wQueue.pop();
    }

    return wResults;
  };

}

#endif // LIB_FLATBUSH_H
