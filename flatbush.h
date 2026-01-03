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

#ifndef FLATBUSH_FLATBUSH_H
#define FLATBUSH_FLATBUSH_H

#include <algorithm>    // for max, min, upper_bound
#include <array>        // for array
#include <cmath>        // for isnan
#include <cstdint>      // for uint32_t, uint8_t, uint16_t, int16_t, int32_t, int8_t
#include <cstring>      // for size_t, memcpy
#include <functional>   // for function
#include <limits>       // for numeric_limits
#include <queue>        // for priority_queue
#include <stdexcept>    // for invalid_argument
#include <string>       // for operator+, to_string, allocator, basic_string, char_traits, string
#include <type_traits>  // for enable_if, is_same, false_type, integral_constant
#include <utility>      // for swap
#include <vector>       // for vector

#define FLATBUSH_USE_AVX512 7
#define FLATBUSH_USE_AVX2 6
#define FLATBUSH_USE_AVX 5
#define FLATBUSH_USE_SSE4 4
#define FLATBUSH_USE_SSSE3 3
#define FLATBUSH_USE_SSE3 2
#define FLATBUSH_USE_SSE2 1

// SIMD intrinsics support detection
#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
#define FLATBUSH_USE_SIMD FLATBUSH_USE_AVX512
#include <immintrin.h>
#pragma message("Detected AVX512 support")
#elif defined(__AVX2__)
#define FLATBUSH_USE_SIMD FLATBUSH_USE_AVX2
#include <immintrin.h>
#pragma message("Detected AVX2 support")
#elif defined(__AVX__)
#define FLATBUSH_USE_SIMD FLATBUSH_USE_AVX
#include <immintrin.h>
#pragma message("Detected AVX support")
#elif defined(__SSE4_1__)
#define FLATBUSH_USE_SIMD FLATBUSH_USE_SSE4
#include <smmintrin.h>
#pragma message("Detected SSE4 support")
#elif defined(__SSSE3__)
#define FLATBUSH_USE_SIMD FLATBUSH_USE_SSSE3
#include <tmmintrin.h>
#pragma message("Detected SSSE3 support")
#elif defined(__SSE3__)
#define FLATBUSH_USE_SIMD FLATBUSH_USE_SSE3
#include <pmmintrin.h>
#pragma message("Detected SSE3 support")
#elif defined(__SSE2__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)))
#define FLATBUSH_USE_SIMD FLATBUSH_USE_SSE2
#include <emmintrin.h>
#pragma message("Detected SSE2 support")
#endif

#if defined(FLATBUSH_USE_SIMD)
#pragma message("Using SIMD intrinsics")
#ifdef _MSC_VER
#include <intrin.h>
#endif
#endif

namespace flatbush {
#ifndef FLATBUSH_SPAN
#include <span>
using std::span;
#else
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
#endif  // FLATBUSH_SPAN

constexpr auto gMaxHilbert = std::numeric_limits<uint16_t>::max();
constexpr auto gMaxDistance = 1.34078e+154;  // std::sqrt(std::numeric_limits<double>::max())
constexpr auto gMaxResults = std::numeric_limits<size_t>::max();
constexpr auto gInvalidArrayType = std::numeric_limits<uint8_t>::max();
constexpr uint16_t gMinNodeSize = 2;
constexpr uint16_t gMaxNodeSize = std::numeric_limits<uint16_t>::max();
constexpr size_t gMaxNumNodes = gMaxNodeSize / 4U;
constexpr size_t gDefaultNodeSize = 16;
constexpr size_t gHeaderByteSize = 8;
constexpr uint8_t gValidityFlag = 0xfb;
constexpr uint8_t gVersion = 3;  // serialized format version

template <typename ArrayType>
struct Box {
  ArrayType mMinX;
  ArrayType mMinY;
  ArrayType mMaxX;
  ArrayType mMaxY;

  template <typename OtherType>
  explicit operator Box<OtherType>() const noexcept {
    return Box<OtherType>{static_cast<OtherType>(mMinX),
                          static_cast<OtherType>(mMinY),
                          static_cast<OtherType>(mMaxX),
                          static_cast<OtherType>(mMaxY)};
  }
};

template <typename ArrayType>
struct Point {
  ArrayType mX;
  ArrayType mY;

  template <typename OtherType>
  explicit operator Point<OtherType>() const noexcept {
    return Point<OtherType>{static_cast<OtherType>(mX), static_cast<OtherType>(mY)};
  }
};

namespace detail {

// From https://www.boost.org/doc/libs/1_81_0/boost/core/bit.hpp (modified)
template <class To, class From>
To bit_cast(From const& from) {
  static_assert(sizeof(To) == sizeof(From), "Cannot cast types of different size");

  To to;
  std::memcpy(&to, &from, sizeof(To));
  return to;
}

inline uint32_t Interleave(uint32_t v) {
  v = (v | (v << 8U)) & 0x00FF00FF;
  v = (v | (v << 4U)) & 0x0F0F0F0F;
  v = (v | (v << 2U)) & 0x33333333;
  v = (v | (v << 1U)) & 0x55555555;
  return v;
}

// From https://github.com/rawrunprotected/hilbert_curves (public domain)
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
  const auto i0 = x ^ y;
  const auto i1 = b | (0xFFFF ^ (i0 | a));

  return (Interleave(i1) << 1U) | Interleave(i0);
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

inline const char* arrayTypeName(size_t iIndex) {
  static constexpr auto kUnknownType = "unknown";
  static constexpr auto kArrayTypeNames = std::array<const char*, 9>{"int8_t",
                                                                     "uint8_t",
                                                                     "uint8_t",
                                                                     "int16_t",
                                                                     "uint16_t",
                                                                     "int32_t",
                                                                     "uint32_t",
                                                                     "float",
                                                                     "double"};
  return iIndex < kArrayTypeNames.size() ? kArrayTypeNames.at(iIndex) : kUnknownType;
}

template <typename ArrayType>
inline size_t approximateResultsSize(const Box<ArrayType>& iBoxIndex,
                                     const Box<ArrayType>& iBoxSearch,
                                     const size_t iNumItems) {
  const auto wBoundsIndex = static_cast<const Box<double>>(iBoxIndex);
  const auto wBoundsSearch = static_cast<const Box<double>>(iBoxSearch);

  // Calculate index area
  const auto wIndexWidth = wBoundsIndex.mMaxX - wBoundsIndex.mMinX;
  const auto wIndexHeight = wBoundsIndex.mMaxY - wBoundsIndex.mMinY;
  const auto wIndexArea = wIndexWidth * wIndexHeight;

  // Calculate intersection area
  const auto wSearchWidth = wBoundsSearch.mMaxX - wBoundsSearch.mMinX;
  const auto wSearchHeight = wBoundsSearch.mMaxY - wBoundsSearch.mMinY;
  const auto wSearchArea = wSearchWidth * wSearchHeight;

  if (wIndexWidth <= 0 || wIndexHeight <= 0 || wIndexArea <= 0 || wSearchWidth <= 0 ||
      wSearchHeight <= 0 || !std::isfinite(wSearchWidth) || !std::isfinite(wSearchHeight) ||
      !std::isfinite(wSearchArea) || wSearchArea <= 0) {
    return 0UL;
  }

  // Approximate results size based as ratio of areas, assuming uniform distribution
  const auto wAreaRatio = wSearchArea / wIndexArea;
  if (!std::isfinite(wAreaRatio) || wAreaRatio > 1.0) {
    return iNumItems;
  }

  return iNumItems * static_cast<size_t>(std::min(1.0, wAreaRatio * 1.8));
}

template <typename ArrayType>
inline bool boxesIntersect(const Box<ArrayType>& iQuery, const Box<ArrayType>& iBox) noexcept {
  return !(iQuery.mMaxX < iBox.mMinX || iQuery.mMaxY < iBox.mMinY || iQuery.mMinX > iBox.mMaxX ||
           iQuery.mMinY > iBox.mMaxY);
}

template <typename ArrayType>
inline void updateBounds(Box<ArrayType>& ioSrc, const Box<ArrayType>& iBox) noexcept {
  ioSrc.mMinX = std::min(ioSrc.mMinX, iBox.mMinX);
  ioSrc.mMinY = std::min(ioSrc.mMinY, iBox.mMinY);
  ioSrc.mMaxX = std::max(ioSrc.mMaxX, iBox.mMaxX);
  ioSrc.mMaxY = std::max(ioSrc.mMaxY, iBox.mMaxY);
}

template <typename ArrayType>
inline double axisDistance(ArrayType iValue, ArrayType iMin, ArrayType iMax) noexcept {
  return std::max(0.0, std::max<double>(iMin - iValue, iValue - iMax));
}

template <typename ArrayType>
inline double computeDistanceSquared(const Point<ArrayType>& iPoint,
                                     const Box<ArrayType>& iBox) noexcept {
  const auto wDistX = axisDistance(iPoint.mX, iBox.mMinX, iBox.mMaxX);
  const auto wDistY = axisDistance(iPoint.mY, iBox.mMinY, iBox.mMaxY);
  return wDistX * wDistX + wDistY * wDistY;
}

#if defined(FLATBUSH_USE_SIMD)
static const auto kZeroPd = _mm_setzero_pd();
static const auto kZeroPs = _mm_setzero_ps();
static const auto kOffset8 = _mm_set1_epi8(std::numeric_limits<int8_t>::min());
static const auto kOffset16 = _mm_set1_epi16(std::numeric_limits<int16_t>::min());
static const auto kOffset32 = _mm_set1_epi32(std::numeric_limits<int32_t>::min());
static const auto kShuffleMin =
    _mm_setr_epi8(0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
static const auto kShuffleMax =
    _mm_setr_epi8(2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
static const auto kMaskInterleave1 = _mm_set1_epi32(0x00FF00FF);
static const auto kMaskInterleave2 = _mm_set1_epi32(0x0F0F0F0F);
static const auto kMaskInterleave3 = _mm_set1_epi32(0x33333333);
static const auto kMaskInterleave4 = _mm_set1_epi32(0x55555555);
static const auto kMaskAllOnes = _mm_set1_epi32(0xFFFF);

#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
static const auto kShuffleMinX512 = _mm512_setr_epi64(0, 4, 8, 12, 0, 0, 0, 0);
static const auto kShuffleMinY512 = _mm512_setr_epi64(1, 5, 9, 13, 0, 0, 0, 0);
static const auto kShuffleMaxX512 = _mm512_setr_epi64(2, 6, 10, 14, 0, 0, 0, 0);
static const auto kShuffleMaxY512 = _mm512_setr_epi64(3, 7, 11, 15, 0, 0, 0, 0);
#endif

inline __m128i Interleave(__m128i v) {
  v = _mm_or_si128(v, _mm_slli_epi32(v, 8));
  v = _mm_and_si128(v, kMaskInterleave1);
  v = _mm_or_si128(v, _mm_slli_epi32(v, 4));
  v = _mm_and_si128(v, kMaskInterleave2);
  v = _mm_or_si128(v, _mm_slli_epi32(v, 2));
  v = _mm_and_si128(v, kMaskInterleave3);
  v = _mm_or_si128(v, _mm_slli_epi32(v, 1));
  v = _mm_and_si128(v, kMaskInterleave4);
  return v;
}

inline __m128i HilbertXYToIndex(__m128i x, __m128i y) {
  // Initial prefix scan round
  auto a = _mm_xor_si128(x, y);
  auto b = _mm_xor_si128(kMaskAllOnes, a);
  auto c = _mm_xor_si128(kMaskAllOnes, _mm_or_si128(x, y));
  auto d = _mm_and_si128(x, _mm_xor_si128(y, kMaskAllOnes));
  auto A = _mm_or_si128(a, _mm_srli_epi32(b, 1));
  auto B = _mm_xor_si128(_mm_srli_epi32(a, 1), a);
  auto C =
      _mm_xor_si128(_mm_xor_si128(_mm_srli_epi32(c, 1), _mm_and_si128(b, _mm_srli_epi32(d, 1))), c);
  auto D =
      _mm_xor_si128(_mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(c, 1)), _mm_srli_epi32(d, 1)), d);

  a = A;
  b = B;
  c = C;
  d = D;
  A = _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(a, 2)), _mm_and_si128(b, _mm_srli_epi32(b, 2)));
  B = _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(b, 2)),
                    _mm_and_si128(b, _mm_srli_epi32(_mm_xor_si128(a, b), 2)));
  C = _mm_xor_si128(C,
                    _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(c, 2)),
                                  _mm_and_si128(b, _mm_srli_epi32(d, 2))));
  D = _mm_xor_si128(D,
                    _mm_xor_si128(_mm_and_si128(b, _mm_srli_epi32(c, 2)),
                                  _mm_and_si128(_mm_xor_si128(a, b), _mm_srli_epi32(d, 2))));

  a = A;
  b = B;
  c = C;
  d = D;
  A = _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(a, 4)), _mm_and_si128(b, _mm_srli_epi32(b, 4)));
  B = _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(b, 4)),
                    _mm_and_si128(b, _mm_srli_epi32(_mm_xor_si128(a, b), 4)));
  C = _mm_xor_si128(C,
                    _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(c, 4)),
                                  _mm_and_si128(b, _mm_srli_epi32(d, 4))));
  D = _mm_xor_si128(D,
                    _mm_xor_si128(_mm_and_si128(b, _mm_srli_epi32(c, 4)),
                                  _mm_and_si128(_mm_xor_si128(a, b), _mm_srli_epi32(d, 4))));

  // Final round
  a = A;
  b = B;
  c = C;
  d = D;
  C = _mm_xor_si128(C,
                    _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(c, 8)),
                                  _mm_and_si128(b, _mm_srli_epi32(d, 8))));
  D = _mm_xor_si128(D,
                    _mm_xor_si128(_mm_and_si128(b, _mm_srli_epi32(c, 8)),
                                  _mm_and_si128(_mm_xor_si128(a, b), _mm_srli_epi32(d, 8))));

  // Undo transformation
  a = _mm_xor_si128(C, _mm_srli_epi32(C, 1));
  b = _mm_xor_si128(D, _mm_srli_epi32(D, 1));

  // Recover index bits and interleave
  const auto i0 = _mm_xor_si128(x, y);
  const auto i1 = _mm_or_si128(b, _mm_xor_si128(kMaskAllOnes, _mm_or_si128(i0, a)));

  return _mm_or_si128(_mm_slli_epi32(Interleave(i1), 1), Interleave(i0));
}

template <>
inline bool boxesIntersect<float>(const Box<float>& iQuery, const Box<float>& iBox) noexcept {
  const auto wQuery = _mm_loadu_ps(&iQuery.mMinX);
  const auto wBox = _mm_loadu_ps(&iBox.mMinX);
  const auto wMin = _mm_unpacklo_ps(wQuery, wBox);
  const auto wMax = _mm_unpackhi_ps(wBox, wQuery);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
  return _mm_cmp_ps_mask(wMax, wMin, _CMP_LT_OQ) == 0;
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  const auto wCmp = _mm_cmplt_ps(wMax, wMin);
  return _mm_testz_ps(wCmp, wCmp);
#else
  return _mm_movemask_ps(_mm_cmplt_ps(wMax, wMin)) == 0;
#endif
}

template <>
inline bool boxesIntersect<double>(const Box<double>& iQuery, const Box<double>& iBox) noexcept {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  const auto wQuery = _mm256_loadu_pd(&iQuery.mMinX);
  const auto wBox = _mm256_loadu_pd(&iBox.mMinX);
  const auto wMax = _mm256_permute2f128_pd(wQuery, wBox, 0x31);
  const auto wMin = _mm256_permute2f128_pd(wBox, wQuery, 0x20);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
  return _mm256_cmp_pd_mask(wMax, wMin, _CMP_LT_OQ) == 0;
#else
  return _mm256_movemask_pd(_mm256_cmp_pd(wMax, wMin, _CMP_LT_OQ)) == 0;
#endif
#else  // if FLATBUSH_USE_SIMD < FLATBUSH_USE_AVX
  const auto wQueryMax = _mm_loadu_pd(&iQuery.mMaxX);
  const auto wBoxMin = _mm_loadu_pd(&iBox.mMinX);
  const auto wCmp1 = _mm_cmplt_pd(wQueryMax, wBoxMin);
  const auto wQueryMin = _mm_loadu_pd(&iQuery.mMinX);
  const auto wBoxMax = _mm_loadu_pd(&iBox.mMaxX);
  const auto wCmp2 = _mm_cmpgt_pd(wQueryMin, wBoxMax);
  return _mm_movemask_pd(_mm_or_pd(wCmp1, wCmp2)) == 0;
#endif
}

template <>
inline bool boxesIntersect<int8_t>(const Box<int8_t>& iQuery, const Box<int8_t>& iBox) noexcept {
  const auto wQuery = _mm_loadu_si32(&iQuery.mMinX);
  const auto wBox = _mm_loadu_si32(&iBox.mMinX);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSSE3
  const auto wMin = _mm_unpacklo_epi16(_mm_shuffle_epi8(wQuery, kShuffleMin),
                                       _mm_shuffle_epi8(wBox, kShuffleMin));
  const auto wMax = _mm_unpacklo_epi16(_mm_shuffle_epi8(wBox, kShuffleMax),
                                       _mm_shuffle_epi8(wQuery, kShuffleMax));
#else
  const auto wMin = _mm_unpacklo_epi8(_mm_shufflelo_epi16(wQuery, _MM_SHUFFLE(0, 0, 0, 0)),
                                      _mm_shufflelo_epi16(wBox, _MM_SHUFFLE(0, 0, 0, 0)));
  const auto wMax = _mm_unpacklo_epi8(_mm_shufflelo_epi16(wBox, _MM_SHUFFLE(1, 1, 1, 1)),
                                      _mm_shufflelo_epi16(wQuery, _MM_SHUFFLE(1, 1, 1, 1)));
#endif
  const auto wCmp = _mm_cmplt_epi8(wMax, wMin);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  return _mm_testz_si128(wCmp, wCmp);
#else
  return _mm_movemask_epi8(wCmp) == 0;
#endif
}

template <>
inline bool boxesIntersect<uint8_t>(const Box<uint8_t>& iQuery, const Box<uint8_t>& iBox) noexcept {
  const auto wQuery = _mm_loadu_si32(&iQuery.mMinX);
  const auto wBox = _mm_loadu_si32(&iBox.mMinX);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSSE3
  const auto wMin = _mm_unpacklo_epi16(_mm_shuffle_epi8(wQuery, kShuffleMin),
                                       _mm_shuffle_epi8(wBox, kShuffleMin));
  const auto wMax = _mm_unpacklo_epi16(_mm_shuffle_epi8(wBox, kShuffleMax),
                                       _mm_shuffle_epi8(wQuery, kShuffleMax));
#else
  const auto wMin = _mm_unpacklo_epi8(_mm_shufflelo_epi16(wQuery, _MM_SHUFFLE(0, 0, 0, 0)),
                                      _mm_shufflelo_epi16(wBox, _MM_SHUFFLE(0, 0, 0, 0)));
  const auto wMax = _mm_unpacklo_epi8(_mm_shufflelo_epi16(wBox, _MM_SHUFFLE(1, 1, 1, 1)),
                                      _mm_shufflelo_epi16(wQuery, _MM_SHUFFLE(1, 1, 1, 1)));
#endif
  const auto wCmp = _mm_cmplt_epi8(_mm_add_epi8(wMax, kOffset8), _mm_add_epi8(wMin, kOffset8));
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  return _mm_testz_si128(wCmp, wCmp);
#else
  return _mm_movemask_epi8(wCmp) == 0;
#endif
}

template <>
inline bool boxesIntersect<int16_t>(const Box<int16_t>& iQuery, const Box<int16_t>& iBox) noexcept {
  const auto wQuery = _mm_loadu_si64(&iQuery.mMinX);
  const auto wBox = _mm_loadu_si64(&iBox.mMinX);
  const auto wMin = _mm_unpacklo_epi16(_mm_shuffle_epi32(wQuery, _MM_SHUFFLE(0, 0, 0, 0)),
                                       _mm_shuffle_epi32(wBox, _MM_SHUFFLE(0, 0, 0, 0)));
  const auto wMax = _mm_unpacklo_epi16(_mm_shuffle_epi32(wBox, _MM_SHUFFLE(1, 1, 1, 1)),
                                       _mm_shuffle_epi32(wQuery, _MM_SHUFFLE(1, 1, 1, 1)));
  const auto wCmp = _mm_cmplt_epi16(wMax, wMin);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  return _mm_testz_si128(wCmp, wCmp);
#else
  return _mm_movemask_epi8(wCmp) == 0;
#endif
}

template <>
inline bool boxesIntersect<uint16_t>(const Box<uint16_t>& iQuery,
                                     const Box<uint16_t>& iBox) noexcept {
  const auto wQuery = _mm_loadu_si64(&iQuery.mMinX);
  const auto wBox = _mm_loadu_si64(&iBox.mMinX);
  const auto wMin = _mm_unpacklo_epi16(_mm_shuffle_epi32(wQuery, _MM_SHUFFLE(0, 0, 0, 0)),
                                       _mm_shuffle_epi32(wBox, _MM_SHUFFLE(0, 0, 0, 0)));
  const auto wMax = _mm_unpacklo_epi16(_mm_shuffle_epi32(wBox, _MM_SHUFFLE(1, 1, 1, 1)),
                                       _mm_shuffle_epi32(wQuery, _MM_SHUFFLE(1, 1, 1, 1)));
  const auto wCmp = _mm_cmplt_epi16(_mm_add_epi16(wMax, kOffset16), _mm_add_epi16(wMin, kOffset16));
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  return _mm_testz_si128(wCmp, wCmp);
#else
  return _mm_movemask_epi8(wCmp) == 0;
#endif
}

template <>
inline bool boxesIntersect<int32_t>(const Box<int32_t>& iQuery, const Box<int32_t>& iBox) noexcept {
  const auto wQuery = _mm_loadu_si128(bit_cast<const __m128i*>(&iQuery.mMinX));
  const auto wBox = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
  const auto wMin = _mm_unpacklo_epi32(wQuery, wBox);
  const auto wMax = _mm_unpackhi_epi32(wBox, wQuery);
  const auto wCmp = _mm_cmplt_epi32(wMax, wMin);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  return _mm_testz_si128(wCmp, wCmp);
#else
  return _mm_movemask_epi8(wCmp) == 0;
#endif
}

template <>
inline bool boxesIntersect<uint32_t>(const Box<uint32_t>& iQuery,
                                     const Box<uint32_t>& iBox) noexcept {
  const auto wQuery = _mm_loadu_si128(bit_cast<const __m128i*>(&iQuery.mMinX));
  const auto wBox = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
  const auto wMin = _mm_unpacklo_epi32(wQuery, wBox);
  const auto wMax = _mm_unpackhi_epi32(wBox, wQuery);
  const auto wCmp = _mm_cmplt_epi32(_mm_add_epi32(wMax, kOffset32), _mm_add_epi32(wMin, kOffset32));
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  return _mm_testz_si128(wCmp, wCmp);
#else
  return _mm_movemask_epi8(wCmp) == 0;
#endif
}

template <>
inline void updateBounds<float>(Box<float>& ioSrc, const Box<float>& iBox) noexcept {
  const auto wCur = _mm_loadu_ps(&ioSrc.mMinX);
  const auto wNew = _mm_loadu_ps(&iBox.mMinX);
  const auto wMins = _mm_min_ps(wCur, wNew);
  const auto wMaxs = _mm_max_ps(wCur, wNew);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  _mm_storeu_ps(&ioSrc.mMinX, _mm_blend_ps(wMins, wMaxs, 0xC));
#else
  _mm_storeu_ps(&ioSrc.mMinX, _mm_shuffle_ps(wMins, wMaxs, _MM_SHUFFLE(3, 2, 1, 0)));
#endif
}

template <>
inline void updateBounds<double>(Box<double>& ioSrc, const Box<double>& iBox) noexcept {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  const auto wCur = _mm256_loadu_pd(&ioSrc.mMinX);
  const auto wNew = _mm256_loadu_pd(&iBox.mMinX);
  const auto wMins = _mm256_min_pd(wCur, wNew);
  const auto wMaxs = _mm256_max_pd(wCur, wNew);
  _mm256_storeu_pd(&ioSrc.mMinX, _mm256_blend_pd(wMins, wMaxs, 0xC));
#else
  _mm_storeu_pd(&ioSrc.mMinX, _mm_min_pd(_mm_loadu_pd(&ioSrc.mMinX), _mm_loadu_pd(&iBox.mMinX)));
  _mm_storeu_pd(&ioSrc.mMaxX, _mm_max_pd(_mm_loadu_pd(&ioSrc.mMaxX), _mm_loadu_pd(&iBox.mMaxX)));
#endif
}

template <>
inline void updateBounds<int8_t>(Box<int8_t>& ioSrc, const Box<int8_t>& iBox) noexcept {
  const auto wCur = _mm_loadu_si32(&ioSrc.mMinX);
  const auto wNew = _mm_loadu_si32(&iBox.mMinX);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wMins = _mm_min_epi8(wCur, wNew);
  const auto wMaxs = _mm_max_epi8(wCur, wNew);
#else
  const auto wCmpMin = _mm_cmplt_epi8(wCur, wNew);
  const auto wCmpMax = _mm_cmpgt_epi8(wCur, wNew);
  const auto wMins = _mm_or_si128(_mm_and_si128(wCmpMin, wCur), _mm_andnot_si128(wCmpMin, wNew));
  const auto wMaxs = _mm_or_si128(_mm_and_si128(wCmpMax, wCur), _mm_andnot_si128(wCmpMax, wNew));
#endif
  _mm_storeu_si32(&ioSrc.mMinX, _mm_unpacklo_epi16(wMins, wMaxs));
}

template <>
inline void updateBounds<uint8_t>(Box<uint8_t>& ioSrc, const Box<uint8_t>& iBox) noexcept {
  const auto wCur = _mm_loadu_si32(&ioSrc.mMinX);
  const auto wNew = _mm_loadu_si32(&iBox.mMinX);
  const auto wMins = _mm_min_epu8(wCur, wNew);
  const auto wMaxs = _mm_max_epu8(wCur, wNew);
  _mm_storeu_si32(&ioSrc.mMinX, _mm_unpacklo_epi16(wMins, wMaxs));
}

template <>
inline void updateBounds<int16_t>(Box<int16_t>& ioSrc, const Box<int16_t>& iBox) noexcept {
  const auto wCur = _mm_loadu_si64(&ioSrc.mMinX);
  const auto wNew = _mm_loadu_si64(&iBox.mMinX);
  const auto wMins = _mm_min_epi16(wCur, wNew);
  const auto wMaxs = _mm_max_epi16(wCur, wNew);
  _mm_storeu_si64(&ioSrc.mMinX, _mm_unpacklo_epi32(wMins, wMaxs));
}

template <>
inline void updateBounds<uint16_t>(Box<uint16_t>& ioSrc, const Box<uint16_t>& iBox) noexcept {
  const auto wCur = _mm_loadu_si64(&ioSrc.mMinX);
  const auto wNew = _mm_loadu_si64(&iBox.mMinX);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wMins = _mm_min_epu16(wCur, wNew);
  const auto wMaxs = _mm_max_epu16(wCur, wNew);
#else
  const auto wCurOff = _mm_add_epi16(wCur, kOffset16);
  const auto wNewOff = _mm_add_epi16(wNew, kOffset16);
  const auto wMins = _mm_sub_epi16(_mm_min_epi16(wCurOff, wNewOff), kOffset16);
  const auto wMaxs = _mm_sub_epi16(_mm_max_epi16(wCurOff, wNewOff), kOffset16);
#endif
  _mm_storeu_si64(&ioSrc.mMinX, _mm_unpacklo_epi32(wMins, wMaxs));
}

template <>
inline void updateBounds<int32_t>(Box<int32_t>& ioSrc, const Box<int32_t>& iBox) noexcept {
  const auto wCurrent = _mm_loadu_si128(bit_cast<const __m128i*>(&ioSrc.mMinX));
  const auto wNewVals = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wMins = _mm_min_epi32(wCurrent, wNewVals);
  const auto wMaxs = _mm_max_epi32(wCurrent, wNewVals);
#else
  const auto wCmpMin = _mm_cmplt_epi32(wCurrent, wNewVals);
  const auto wCmpMax = _mm_cmpgt_epi32(wCurrent, wNewVals);
  const auto wMins =
      _mm_or_si128(_mm_and_si128(wCmpMin, wCurrent), _mm_andnot_si128(wCmpMin, wNewVals));
  const auto wMaxs =
      _mm_or_si128(_mm_and_si128(wCmpMax, wCurrent), _mm_andnot_si128(wCmpMax, wNewVals));
#endif
  _mm_storeu_si128(bit_cast<__m128i*>(&ioSrc.mMinX), _mm_unpacklo_epi64(wMins, wMaxs));
}

template <>
inline void updateBounds<uint32_t>(Box<uint32_t>& ioSrc, const Box<uint32_t>& iBox) noexcept {
  const auto wCur = _mm_loadu_si128(bit_cast<const __m128i*>(&ioSrc.mMinX));
  const auto wNew = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wMins = _mm_min_epu32(wCur, wNew);
  const auto wMaxs = _mm_max_epu32(wCur, wNew);
#else
  const auto wCurOff = _mm_add_epi32(wCur, kOffset32);
  const auto wNewOff = _mm_add_epi32(wNew, kOffset32);
  const auto wCmpMin = _mm_cmplt_epi32(wCurOff, wNewOff);
  const auto wCmpMax = _mm_cmpgt_epi32(wCurOff, wNewOff);
  const auto wMins = _mm_sub_epi32(
      _mm_or_si128(_mm_and_si128(wCmpMin, wCurOff), _mm_andnot_si128(wCmpMin, wNewOff)), kOffset32);
  const auto wMaxs = _mm_sub_epi32(
      _mm_or_si128(_mm_and_si128(wCmpMax, wCurOff), _mm_andnot_si128(wCmpMax, wNewOff)), kOffset32);
#endif
  _mm_storeu_si128(bit_cast<__m128i*>(&ioSrc.mMinX), _mm_unpacklo_epi64(wMins, wMaxs));
}

template <>
inline double computeDistanceSquared<double>(const Point<double>& iPoint,
                                             const Box<double>& iBox) noexcept {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  const auto wBox = _mm256_loadu_pd(&iBox.mMinX);
  const auto wBoxMin = _mm256_castpd256_pd128(wBox);
  const auto wBoxMax = _mm256_extractf128_pd(wBox, 1);
#else
  const auto wBoxMin = _mm_loadu_pd(&iBox.mMinX);
  const auto wBoxMax = _mm_loadu_pd(&iBox.mMaxX);
#endif
  const auto wPoint = _mm_loadu_pd(&iPoint.mX);
  // Compute axis distances - using max to clamp to zero
  const auto wDist =
      _mm_max_pd(kZeroPd, _mm_max_pd(_mm_sub_pd(wBoxMin, wPoint), _mm_sub_pd(wPoint, wBoxMax)));
  // Square and sum
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wResult = _mm_dp_pd(wDist, wDist, 0x31);
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE3
  const auto wDistSq = _mm_mul_pd(wDist, wDist);
  const auto wResult = _mm_hadd_pd(wDistSq, wDistSq);
#else
  const auto wDistSq = _mm_mul_pd(wDist, wDist);
  const auto wResult = _mm_add_pd(wDistSq, _mm_unpackhi_pd(wDistSq, wDistSq));
#endif
  return _mm_cvtsd_f64(wResult);
}

template <>
inline double computeDistanceSquared<float>(const Point<float>& iPoint,
                                            const Box<float>& iBox) noexcept {
  const auto wPoint = _mm_castpd_ps(_mm_load_sd(bit_cast<const double*>(&iPoint.mX)));
  const auto wPointHl = _mm_movelh_ps(wPoint, wPoint);
  const auto wBox = _mm_loadu_ps(&iBox.mMinX);
  const auto wBoxMin = _mm_movelh_ps(wBox, wBox);
  const auto wBoxMax = _mm_movehl_ps(wBox, wBox);
  // Compute axis distances - using max to clamp to zero
  const auto wDist =
      _mm_max_ps(kZeroPs, _mm_max_ps(_mm_sub_ps(wBoxMin, wPointHl), _mm_sub_ps(wPointHl, wBoxMax)));
  // Square and sum
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wResult = _mm_dp_ps(wDist, wDist, 0x31);
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE3
  const auto wDistSq = _mm_mul_ps(wDist, wDist);
  const auto wResult = _mm_hadd_ps(wDistSq, wDistSq);
#else
  const auto wDistSq = _mm_mul_ps(wDist, wDist);
  const auto wShuf = _mm_shuffle_ps(wDistSq, wDistSq, _MM_SHUFFLE(1, 1, 1, 1));
  const auto wResult = _mm_add_ps(wDistSq, wShuf);
#endif
  return static_cast<double>(_mm_cvtss_f32(wResult));
}

template <>
inline double computeDistanceSquared<int8_t>(const Point<int8_t>& iPoint,
                                             const Box<int8_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}

template <>
inline double computeDistanceSquared<uint8_t>(const Point<uint8_t>& iPoint,
                                              const Box<uint8_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}

template <>
inline double computeDistanceSquared<int16_t>(const Point<int16_t>& iPoint,
                                              const Box<int16_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}

template <>
inline double computeDistanceSquared<uint16_t>(const Point<uint16_t>& iPoint,
                                               const Box<uint16_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}

template <>
inline double computeDistanceSquared<int32_t>(const Point<int32_t>& iPoint,
                                              const Box<int32_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}

template <>
inline double computeDistanceSquared<uint32_t>(const Point<uint32_t>& iPoint,
                                               const Box<uint32_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}

void loadBoxValuesAsFloat(
    const Box<float>& iBox, __m128& oMinX, __m128& oMinY, __m128& oMaxX, __m128& oMaxY) {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
  const auto wData = _mm512_loadu_ps(&iBox.mMinX);
  oMinX = _mm512_castps512_ps128(wData);
  oMinY = _mm512_extractf32x4_ps(wData, 1);
  oMaxX = _mm512_extractf32x4_ps(wData, 2);
  oMaxY = _mm512_extractf32x4_ps(wData, 3);
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  const auto wData1 = _mm256_loadu_ps(&iBox.mMinX);
  const auto wData2 = _mm256_loadu_ps(&iBox.mMinX + 8);
  oMinX = _mm256_castps256_ps128(wData1);
  oMinY = _mm256_extractf128_ps(wData1, 1);
  oMaxX = _mm256_castps256_ps128(wData2);
  oMaxY = _mm256_extractf128_ps(wData2, 1);
#else
  oMinX = _mm_loadu_ps(&iBox.mMinX);
  oMinY = _mm_loadu_ps(&iBox.mMinX + 4);
  oMaxX = _mm_loadu_ps(&iBox.mMinX + 8);
  oMaxY = _mm_loadu_ps(&iBox.mMinX + 12);
#endif
}

void loadBoxValuesAsFloat(
    const Box<int8_t>& iBox, __m128& oMinX, __m128& oMinY, __m128& oMaxX, __m128& oMaxY) {
  const auto wData = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wBox0 = _mm_cvtepi8_epi32(wData);
  const auto wBox1 = _mm_cvtepi8_epi32(_mm_srli_si128(wData, 4));
  const auto wBox2 = _mm_cvtepi8_epi32(_mm_srli_si128(wData, 8));
  const auto wBox3 = _mm_cvtepi8_epi32(_mm_srli_si128(wData, 12));
#else
  const auto wData16LoSigned = _mm_srai_epi16(_mm_unpacklo_epi8(wData, wData), 8);
  const auto wBox0 = _mm_srai_epi32(_mm_unpacklo_epi16(wData16LoSigned, wData16LoSigned), 16);
  const auto wBox1 = _mm_srai_epi32(_mm_unpackhi_epi16(wData16LoSigned, wData16LoSigned), 16);
  const auto wData16HiSigned = _mm_srai_epi16(_mm_unpackhi_epi8(wData, wData), 8);
  const auto wBox2 = _mm_srai_epi32(_mm_unpacklo_epi16(wData16HiSigned, wData16HiSigned), 16);
  const auto wBox3 = _mm_srai_epi32(_mm_unpackhi_epi16(wData16HiSigned, wData16HiSigned), 16);
#endif
  oMinX = _mm_cvtepi32_ps(wBox0);
  oMinY = _mm_cvtepi32_ps(wBox1);
  oMaxX = _mm_cvtepi32_ps(wBox2);
  oMaxY = _mm_cvtepi32_ps(wBox3);
}

void loadBoxValuesAsFloat(
    const Box<uint8_t>& iBox, __m128& oMinX, __m128& oMinY, __m128& oMaxX, __m128& oMaxY) {
  const auto wData = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wBox0 = _mm_cvtepi8_epi32(wData);
  const auto wBox1 = _mm_cvtepi8_epi32(_mm_srli_si128(wData, 4));
  const auto wBox2 = _mm_cvtepi8_epi32(_mm_srli_si128(wData, 8));
  const auto wBox3 = _mm_cvtepi8_epi32(_mm_srli_si128(wData, 12));
#else
  const auto wData16LoSigned = _mm_srai_epi16(_mm_unpacklo_epi8(wData, wData), 8);
  const auto wBox0 = _mm_srai_epi32(_mm_unpacklo_epi16(wData16LoSigned, wData16LoSigned), 16);
  const auto wBox1 = _mm_srai_epi32(_mm_unpackhi_epi16(wData16LoSigned, wData16LoSigned), 16);
  const auto wData16HiSigned = _mm_srai_epi16(_mm_unpackhi_epi8(wData, wData), 8);
  const auto wBox2 = _mm_srai_epi32(_mm_unpacklo_epi16(wData16HiSigned, wData16HiSigned), 16);
  const auto wBox3 = _mm_srai_epi32(_mm_unpackhi_epi16(wData16HiSigned, wData16HiSigned), 16);
#endif
  oMinX = _mm_cvtepi32_ps(wBox0);
  oMinY = _mm_cvtepi32_ps(wBox1);
  oMaxX = _mm_cvtepi32_ps(wBox2);
  oMaxY = _mm_cvtepi32_ps(wBox3);
}

void loadBoxValuesAsFloat(
    const Box<int16_t>& iBox, __m128& oMinX, __m128& oMinY, __m128& oMaxX, __m128& oMaxY) {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX2
  const auto wData = _mm256_loadu_si256(bit_cast<const __m256i*>(&iBox.mMinX));
  const auto wData32Lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(wData));
  const auto wData32Hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(wData, 1));
  const auto wBox0 = _mm256_castsi256_si128(wData32Lo);
  const auto wBox1 = _mm256_extracti128_si256(wData32Lo, 1);
  const auto wBox2 = _mm256_castsi256_si128(wData32Hi);
  const auto wBox3 = _mm256_extracti128_si256(wData32Hi, 1);
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wDataLo = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
  const auto wDataHi = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX + 8));
  const auto wBox0 = _mm_cvtepi16_epi32(wDataLo);
  const auto wBox1 = _mm_cvtepi16_epi32(_mm_srli_si128(wDataLo, 8));
  const auto wBox2 = _mm_cvtepi16_epi32(wDataHi);
  const auto wBox3 = _mm_cvtepi16_epi32(_mm_srli_si128(wDataHi, 8));
#else
  const auto wDataLo = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
  const auto wDataHi = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX + 8));
  const auto wBox0 = _mm_srai_epi32(_mm_unpacklo_epi16(wDataLo, wDataLo), 16);
  const auto wBox1 = _mm_srai_epi32(_mm_unpackhi_epi16(wDataLo, wDataLo), 16);
  const auto wBox2 = _mm_srai_epi32(_mm_unpacklo_epi16(wDataHi, wDataHi), 16);
  const auto wBox3 = _mm_srai_epi32(_mm_unpackhi_epi16(wDataHi, wDataHi), 16);
#endif
  oMinX = _mm_cvtepi32_ps(wBox0);
  oMinY = _mm_cvtepi32_ps(wBox1);
  oMaxX = _mm_cvtepi32_ps(wBox2);
  oMaxY = _mm_cvtepi32_ps(wBox3);
}

void loadBoxValuesAsFloat(
    const Box<uint16_t>& iBox, __m128& oMinX, __m128& oMinY, __m128& oMaxX, __m128& oMaxY) {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX2
  const auto wData = _mm256_loadu_si256(bit_cast<const __m256i*>(&iBox.mMinX));
  const auto wData32Lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(wData));
  const auto wData32Hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(wData, 1));
  const auto wBox0 = _mm256_castsi256_si128(wData32Lo);
  const auto wBox1 = _mm256_extracti128_si256(wData32Lo, 1);
  const auto wBox2 = _mm256_castsi256_si128(wData32Hi);
  const auto wBox3 = _mm256_extracti128_si256(wData32Hi, 1);
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wDataLo = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
  const auto wDataHi = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX + 8));
  const auto wBox0 = _mm_cvtepi16_epi32(wDataLo);
  const auto wBox1 = _mm_cvtepi16_epi32(_mm_srli_si128(wDataLo, 8));
  const auto wBox2 = _mm_cvtepi16_epi32(wDataHi);
  const auto wBox3 = _mm_cvtepi16_epi32(_mm_srli_si128(wDataHi, 8));
#else
  const auto wDataLo = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
  const auto wDataHi = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX + 8));
  const auto wBox0 = _mm_srai_epi32(_mm_unpacklo_epi16(wDataLo, wDataLo), 16);
  const auto wBox1 = _mm_srai_epi32(_mm_unpackhi_epi16(wDataLo, wDataLo), 16);
  const auto wBox2 = _mm_srai_epi32(_mm_unpacklo_epi16(wDataHi, wDataHi), 16);
  const auto wBox3 = _mm_srai_epi32(_mm_unpackhi_epi16(wDataHi, wDataHi), 16);
#endif
  oMinX = _mm_cvtepi32_ps(wBox0);
  oMinY = _mm_cvtepi32_ps(wBox1);
  oMaxX = _mm_cvtepi32_ps(wBox2);
  oMaxY = _mm_cvtepi32_ps(wBox3);
}

void loadBoxValuesAsFloat(
    const Box<int32_t>& iBox, __m128& oMinX, __m128& oMinY, __m128& oMaxX, __m128& oMaxY) {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
  const auto wData = _mm512_loadu_si512(bit_cast<const __m512i*>(&iBox.mMinX));
  const auto wBox0 = _mm512_castsi512_si128(wData);
  const auto wBox1 = _mm512_extracti32x4_epi32(wData, 1);
  const auto wBox2 = _mm512_extracti32x4_epi32(wData, 2);
  const auto wBox3 = _mm512_extracti32x4_epi32(wData, 3);
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX2
  const auto wDataLo = _mm256_loadu_si256(bit_cast<const __m256i*>(&iBox.mMinX));
  const auto wDataHi = _mm256_loadu_si256(bit_cast<const __m256i*>(&iBox.mMinX + 8));
  const auto wBox0 = _mm256_castsi256_si128(wDataLo);
  const auto wBox1 = _mm256_extracti128_si256(wDataLo, 1);
  const auto wBox2 = _mm256_castsi256_si128(wDataHi);
  const auto wBox3 = _mm256_extracti128_si256(wDataHi, 1);
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE2
  const auto wBox0 = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
  const auto wBox1 = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX + 4));
  const auto wBox2 = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX + 8));
  const auto wBox3 = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX + 12));
#endif
  oMinX = _mm_cvtepi32_ps(wBox0);
  oMinY = _mm_cvtepi32_ps(wBox1);
  oMaxX = _mm_cvtepi32_ps(wBox2);
  oMaxY = _mm_cvtepi32_ps(wBox3);
}

void loadBoxValuesAsFloat(
    const Box<uint32_t>& iBox, __m128& oMinX, __m128& oMinY, __m128& oMaxX, __m128& oMaxY) {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
  const auto wData = _mm512_loadu_si512(bit_cast<const __m512i*>(&iBox.mMinX));
  const auto wBox0 = _mm512_castsi512_si128(wData);
  const auto wBox1 = _mm512_extracti32x4_epi32(wData, 1);
  const auto wBox2 = _mm512_extracti32x4_epi32(wData, 2);
  const auto wBox3 = _mm512_extracti32x4_epi32(wData, 3);
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX2
  const auto wDataLo = _mm256_loadu_si256(bit_cast<const __m256i*>(&iBox.mMinX));
  const auto wDataHi = _mm256_loadu_si256(bit_cast<const __m256i*>(&iBox.mMinX + 8));
  const auto wBox0 = _mm256_castsi256_si128(wDataLo);
  const auto wBox1 = _mm256_extracti128_si256(wDataLo, 1);
  const auto wBox2 = _mm256_castsi256_si128(wDataHi);
  const auto wBox3 = _mm256_extracti128_si256(wDataHi, 1);
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE2
  const auto wBox0 = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX));
  const auto wBox1 = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX + 4));
  const auto wBox2 = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX + 8));
  const auto wBox3 = _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX + 12));
#endif
  oMinX = _mm_cvtepi32_ps(wBox0);
  oMinY = _mm_cvtepi32_ps(wBox1);
  oMaxX = _mm_cvtepi32_ps(wBox2);
  oMaxY = _mm_cvtepi32_ps(wBox3);
}
#endif  // defined(FLATBUSH_USE_SIMD)

template <class ArrayType>
std::vector<uint32_t> computeHilbertValues(size_t iNumItems,
                                           const Box<ArrayType>& iBounds,
                                           span<Box<ArrayType>> iBoxes) {
  static constexpr auto kMaxHilbertRatio = 0.5f * std::numeric_limits<uint16_t>::max();
  const auto wHilbertWidth = kMaxHilbertRatio / static_cast<float>(iBounds.mMaxX - iBounds.mMinX);
  const auto wHilbertHeight = kMaxHilbertRatio / static_cast<float>(iBounds.mMaxY - iBounds.mMinY);
  const auto wDoubleMinX = static_cast<float>(iBounds.mMinX + iBounds.mMinX);
  const auto wDoubleMinY = static_cast<float>(iBounds.mMinY + iBounds.mMinY);
  auto wHilbertValues = std::vector<uint32_t>(iNumItems);
  auto wIdx = 0UL;

#if defined(FLATBUSH_USE_SIMD)
  const auto wHilbertWidth128 = _mm_set1_ps(wHilbertWidth);
  const auto wHilbertHeight128 = _mm_set1_ps(wHilbertHeight);
  const auto wDoubleMinX128 = _mm_set1_ps(wDoubleMinX);
  const auto wDoubleMinY128 = _mm_set1_ps(wDoubleMinY);

  for (; wIdx + 3 < iNumItems; wIdx += 4) {
    __m128 wMinX;
    __m128 wMinY;
    __m128 wMaxX;
    __m128 wMaxY;
    loadBoxValuesAsFloat(iBoxes[wIdx], wMinX, wMinY, wMaxX, wMaxY);
    _MM_TRANSPOSE4_PS(wMinX, wMinY, wMaxX, wMaxY);
    const auto wSumX = _mm_add_ps(wMinX, wMaxX);
    const auto wSumY = _mm_add_ps(wMinY, wMaxY);
    const auto wResultX = _mm_mul_ps(wHilbertWidth128, _mm_sub_ps(wSumX, wDoubleMinX128));
    const auto wResultY = _mm_mul_ps(wHilbertHeight128, _mm_sub_ps(wSumY, wDoubleMinY128));
    _mm_storeu_si128(bit_cast<__m128i*>(&wHilbertValues[wIdx]),
                     HilbertXYToIndex(_mm_cvtps_epi32(wResultX), _mm_cvtps_epi32(wResultY)));
  }
#endif  // defined(FLATBUSH_USE_SIMD)

  for (; wIdx < iNumItems; ++wIdx) {
    const auto& wBox = static_cast<Box<float>>(iBoxes[wIdx]);
    wHilbertValues[wIdx] = HilbertXYToIndex(
        static_cast<uint32_t>(wHilbertWidth * (wBox.mMinX + wBox.mMaxX - wDoubleMinX)),
        static_cast<uint32_t>(wHilbertHeight * (wBox.mMinY + wBox.mMaxY - wDoubleMinY)));
  }

  return wHilbertValues;
}

template <>
inline std::vector<uint32_t> computeHilbertValues<double>(size_t iNumItems,
                                                          const Box<double>& iBounds,
                                                          span<Box<double>> iBoxes) {
  static constexpr auto kMaxHilbertRatio = 0.5 * std::numeric_limits<uint16_t>::max();
  const auto wHilbertWidth = kMaxHilbertRatio / (iBounds.mMaxX - iBounds.mMinX);
  const auto wHilbertHeight = kMaxHilbertRatio / (iBounds.mMaxY - iBounds.mMinY);
  const auto wDoubleMinX = iBounds.mMinX + iBounds.mMinX;
  const auto wDoubleMinY = iBounds.mMinY + iBounds.mMinY;
  auto wHilbertValues = std::vector<uint32_t>(iNumItems);
  auto wIdx = 0UL;

#if defined(FLATBUSH_USE_SIMD)
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  const auto wHilbertWidth256 = _mm256_set1_pd(wHilbertWidth);
  const auto wHilbertHeight256 = _mm256_set1_pd(wHilbertHeight);
  const auto wDoubleMinX256 = _mm256_set1_pd(wDoubleMinX);
  const auto wDoubleMinY256 = _mm256_set1_pd(wDoubleMinY);

  for (; wIdx + 3 < iNumItems; wIdx += 4) {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
    const auto wBoxes01 = _mm512_loadu_pd(&iBoxes[wIdx].mMinX);
    const auto wBoxes23 = _mm512_loadu_pd(&iBoxes[wIdx + 2].mMinX);
    const auto wMinX =
        _mm512_castpd512_pd256(_mm512_permutex2var_pd(wBoxes01, kShuffleMinX512, wBoxes23));
    const auto wMinY =
        _mm512_castpd512_pd256(_mm512_permutex2var_pd(wBoxes01, kShuffleMinY512, wBoxes23));
    const auto wMaxX =
        _mm512_castpd512_pd256(_mm512_permutex2var_pd(wBoxes01, kShuffleMaxX512, wBoxes23));
    const auto wMaxY =
        _mm512_castpd512_pd256(_mm512_permutex2var_pd(wBoxes01, kShuffleMaxY512, wBoxes23));
#else
    const auto wBox0 = _mm256_loadu_pd(&iBoxes[wIdx].mMinX);
    const auto wBox1 = _mm256_loadu_pd(&iBoxes[wIdx + 1].mMinX);
    const auto wBox2 = _mm256_loadu_pd(&iBoxes[wIdx + 2].mMinX);
    const auto wBox3 = _mm256_loadu_pd(&iBoxes[wIdx + 3].mMinX);
    const auto wBoxes01Lo = _mm256_unpacklo_pd(wBox0, wBox1);
    const auto wBoxes01Hi = _mm256_unpackhi_pd(wBox0, wBox1);
    const auto wBoxes23Lo = _mm256_unpacklo_pd(wBox2, wBox3);
    const auto wBoxes23Hi = _mm256_unpackhi_pd(wBox2, wBox3);
    const auto wMinX = _mm256_permute2f128_pd(wBoxes01Lo, wBoxes23Lo, 0x20);
    const auto wMinY = _mm256_permute2f128_pd(wBoxes01Hi, wBoxes23Hi, 0x20);
    const auto wMaxX = _mm256_permute2f128_pd(wBoxes01Lo, wBoxes23Lo, 0x31);
    const auto wMaxY = _mm256_permute2f128_pd(wBoxes01Hi, wBoxes23Hi, 0x31);
#endif
    const auto wSumX = _mm256_add_pd(wMinX, wMaxX);
    const auto wSumY = _mm256_add_pd(wMinY, wMaxY);
    const auto wResultX = _mm256_mul_pd(wHilbertWidth256, _mm256_sub_pd(wSumX, wDoubleMinX256));
    const auto wResultY = _mm256_mul_pd(wHilbertHeight256, _mm256_sub_pd(wSumY, wDoubleMinY256));

    _mm_storeu_si128(bit_cast<__m128i*>(&wHilbertValues[wIdx]),
                     HilbertXYToIndex(_mm256_cvtpd_epi32(wResultX), _mm256_cvtpd_epi32(wResultY)));
  }
#endif

#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE2
  const auto wHilbertWidth128 = _mm_set1_pd(wHilbertWidth);
  const auto wHilbertHeight128 = _mm_set1_pd(wHilbertHeight);
  const auto wDoubleMinX128 = _mm_set1_pd(wDoubleMinX);
  const auto wDoubleMinY128 = _mm_set1_pd(wDoubleMinY);

  for (; wIdx + 1 < iNumItems; wIdx += 2) {
    const auto wBox0Lo = _mm_loadu_pd(&iBoxes[wIdx].mMinX);
    const auto wBox0Hi = _mm_loadu_pd(&iBoxes[wIdx].mMaxX);
    const auto wBox1Lo = _mm_loadu_pd(&iBoxes[wIdx + 1].mMinX);
    const auto wBox1Hi = _mm_loadu_pd(&iBoxes[wIdx + 1].mMaxX);

    const auto wMinX = _mm_unpacklo_pd(wBox0Lo, wBox1Lo);
    const auto wMinY = _mm_unpackhi_pd(wBox0Lo, wBox1Lo);
    const auto wMaxX = _mm_unpacklo_pd(wBox0Hi, wBox1Hi);
    const auto wMaxY = _mm_unpackhi_pd(wBox0Hi, wBox1Hi);

    const auto wSumX = _mm_add_pd(wMinX, wMaxX);
    const auto wSumY = _mm_add_pd(wMinY, wMaxY);
    const auto wResultX = _mm_mul_pd(wHilbertWidth128, _mm_sub_pd(wSumX, wDoubleMinX128));
    const auto wResultY = _mm_mul_pd(wHilbertHeight128, _mm_sub_pd(wSumY, wDoubleMinY128));

    _mm_storeu_si64(bit_cast<__m128i*>(&wHilbertValues[wIdx]),
                    HilbertXYToIndex(_mm_cvtpd_epi32(wResultX), _mm_cvtpd_epi32(wResultY)));
  }
#endif  // FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE2
#endif  // defined(FLATBUSH_USE_SIMD)

  for (; wIdx < iNumItems; ++wIdx) {
    const auto& wBox = iBoxes[wIdx];
    wHilbertValues.at(wIdx) =
        HilbertXYToIndex(uint32_t(wHilbertWidth * (wBox.mMinX + wBox.mMaxX - wDoubleMinX)),
                         uint32_t(wHilbertHeight * (wBox.mMinY + wBox.mMaxY - wDoubleMinY)));
  }

  return wHilbertValues;
}
}  // namespace detail

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

  inline size_t add(Box<ArrayType>&& iBox) noexcept {
    mItems.push_back(std::move(iBox));
    return mItems.size() - 1UL;
  }

  Flatbush<ArrayType> finish();
  static Flatbush<ArrayType> from(const uint8_t* iData, size_t iSize);
  static Flatbush<ArrayType> from(std::vector<uint8_t>&& iData);

 private:
  static void validate(const uint8_t* iData, size_t iSize);
  std::uint16_t mNodeSize;
  std::vector<Box<ArrayType>> mItems;
};

template <typename ArrayType>
Flatbush<ArrayType> FlatbushBuilder<ArrayType>::finish() {
  if (mItems.empty()) {
    throw std::invalid_argument("No items have been added. Nothing to build.");
  }

  Flatbush<ArrayType> wIndex(uint32_t(mItems.size()), mNodeSize);
  wIndex.create(std::move(mItems));

  return wIndex;
}

template <typename ArrayType>
Flatbush<ArrayType> FlatbushBuilder<ArrayType>::from(const uint8_t* iData, size_t iSize) {
  validate(iData, iSize);

  auto wInstance = Flatbush<ArrayType>(iData, iSize);
  const auto wSize = wInstance.data().size();

  if (wSize != iSize) {
    throw std::invalid_argument("Num items dictates a total size of " + std::to_string(wSize) +
                                ", but got buffer size " + std::to_string(iSize) + ".");
  }

  return wInstance;
}

template <typename ArrayType>
Flatbush<ArrayType> FlatbushBuilder<ArrayType>::from(std::vector<uint8_t>&& iData) {
  validate(iData.data(), iData.size());

  const auto wDataSize = iData.size();
  auto wInstance = Flatbush<ArrayType>(std::move(iData));
  const auto wSize = wInstance.data().size();

  if (wSize != wDataSize) {
    throw std::invalid_argument("Num items dictates a total size of " + std::to_string(wSize) +
                                ", but got buffer size " + std::to_string(wDataSize) + ".");
  }

  return wInstance;
}

template <typename ArrayType>
void FlatbushBuilder<ArrayType>::validate(const uint8_t* iData, size_t iSize) {
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
    throw std::invalid_argument(std::string("Expected type is ")
                                    .append(detail::arrayTypeName(wEncodedType))
                                    .append(", but got template type ")
                                    .append(detail::arrayTypeName(wExpectedType)));
  }

  const auto wNodeSize = *detail::bit_cast<const uint16_t*>(&iData[2]);
  if (wNodeSize < gMinNodeSize) {
    throw std::invalid_argument("Node size cannot be < " + std::to_string(gMinNodeSize) + ".");
  }
}

template <typename ArrayType>
class Flatbush {
  using FilterCb = std::function<bool(size_t, const Box<ArrayType>&)>;

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

  inline size_t nodeSize() const noexcept { return *detail::bit_cast<const uint16_t*>(&mData[2]); };

  inline size_t numItems() const noexcept { return *detail::bit_cast<const uint32_t*>(&mData[4]); };

  inline size_t indexSize() const noexcept { return mBoxes.size(); };

  inline span<const uint8_t> data() const noexcept { return {mData.data(), mData.capacity()}; };

  friend class FlatbushBuilder<ArrayType>;

 private:
  static constexpr ArrayType cMaxValue = std::numeric_limits<ArrayType>::max();
  static constexpr ArrayType cMinValue = std::numeric_limits<ArrayType>::lowest();

  static inline double axisDistance(ArrayType iValue, ArrayType iMin, ArrayType iMax) noexcept {
    return iValue < iMin ? iMin - iValue : std::max<double>(iValue - iMax, 0.0);
  }

  inline bool canDoSearch(const Box<ArrayType>& iBounds) const {
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

    return !wIsNanBounds && iBounds.mMaxX >= mBounds.mMinX && iBounds.mMinX <= mBounds.mMaxX &&
           iBounds.mMaxY >= mBounds.mMinY && iBounds.mMinY <= mBounds.mMaxY;
  }

  inline bool canDoNeighbors(const Point<ArrayType>& iPoint,
                             size_t iMaxResults,
                             double iMaxDistance,
                             double iMaxDistSquared) const {
#if defined(_WIN32) || defined(_WIN64)
    // On Windows, isnan throws on anything that is not float, double or long double
    const auto wIsNanPoint =
        (std::isnan(static_cast<double>(iPoint.mX)) || std::isnan(static_cast<double>(iPoint.mY)));
#else
    const auto wIsNanPoint = (std::isnan(iPoint.mX) || std::isnan(iPoint.mY));
#endif

    const auto wDistSquared = detail::computeDistanceSquared(iPoint, mBounds);

    return !wIsNanPoint && iMaxResults != 0UL && iMaxDistance > 0.0 && !std::isnan(wDistSquared) &&
           std::isnormal(iMaxDistSquared) && wDistSquared <= iMaxDistSquared;
  }

  Flatbush(uint32_t iNumItems, uint16_t iNodeSize) noexcept;
  Flatbush(const uint8_t* iData, size_t iSize) noexcept;
  explicit Flatbush(std::vector<uint8_t>&& iData) noexcept;

  void create(std::vector<Box<ArrayType>>&& iItems) noexcept;
  void init(uint32_t iNumItems, uint32_t iNodeSize) noexcept;
  uint32_t medianOfThree(const std::vector<uint32_t>& iValues,
                         size_t iLeft,
                         size_t iRight) noexcept;
  void sort(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight) noexcept;
  void swap(std::vector<uint32_t>& iValues, size_t iLeft, size_t iRight) noexcept;
  inline size_t upperBound(size_t iNodeIndex) const noexcept;

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
  mData[0] = gValidityFlag;
  mData[1] = (gVersion << 4U) + detail::arrayTypeIndex<ArrayType>();
  *detail::bit_cast<uint16_t*>(&mData[2]) = iNodeSize;
  *detail::bit_cast<uint32_t*>(&mData[4]) = iNumItems;
}

template <typename ArrayType>
Flatbush<ArrayType>::Flatbush(const uint8_t* iData, size_t iSize) noexcept {
  const auto wNodeSize = *detail::bit_cast<const uint16_t*>(&iData[2]);
  const auto wNumItems = *detail::bit_cast<const uint32_t*>(&iData[4]);
  mData.insert(mData.begin(), iData, iData + iSize);
  init(wNumItems, wNodeSize);
  mPosition = mLevelBounds.empty() ? 0UL : mLevelBounds.back();

  if (mPosition > 0UL) {
    mBounds = mBoxes[mPosition - 1UL];
  }
}

template <typename ArrayType>
Flatbush<ArrayType>::Flatbush(std::vector<uint8_t>&& iData) noexcept {
  const auto wNodeSize = *detail::bit_cast<const uint16_t*>(&iData.at(2));
  const auto wNumItems = *detail::bit_cast<const uint32_t*>(&iData.at(4));
  mData = std::move(iData);
  init(wNumItems, wNodeSize);
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
  mData.resize(wDataSize, 0U);
  mBoxes = {detail::bit_cast<Box<ArrayType>*>(&mData[gHeaderByteSize]), wNumNodes};
  mIndicesUint16 = {detail::bit_cast<uint16_t*>(&mData[gHeaderByteSize + wNodesByteSize]),
                    wNumNodes};
  mIndicesUint32 = {detail::bit_cast<uint32_t*>(&mData[gHeaderByteSize + wNodesByteSize]),
                    wNumNodes};
}

template <typename ArrayType>
void Flatbush<ArrayType>::create(std::vector<Box<ArrayType>>&& iItems) noexcept {
  for (auto&& wBox : iItems) {
    if (mIsWideIndex) {
      mIndicesUint32[mPosition] = uint32_t(mPosition);
    } else {
      mIndicesUint16[mPosition] = uint16_t(mPosition);
    }

    mBoxes[mPosition] = std::move(wBox);
    detail::updateBounds(mBounds, mBoxes[mPosition]);
    ++mPosition;
  }

  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();

  if (wNumItems <= wNodeSize) {
    mBoxes[mPosition++] = mBounds;
    return;
  }

  // map item centers into Hilbert coordinate space and calculate Hilbert values
  auto wHilbertValues = detail::computeHilbertValues(wNumItems, mBounds, mBoxes);
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
        detail::updateBounds(wNodeBox, mBoxes[wPosition]);
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

template <typename ArrayType>
uint32_t Flatbush<ArrayType>::medianOfThree(const std::vector<uint32_t>& iValues,
                                            size_t iLeft,
                                            size_t iRight) noexcept {
  const auto wStart = iValues.at(iLeft);
  const auto wMid = iValues.at((iLeft + iRight) >> 1);
  const auto wEnd = iValues.at(iRight);
  const auto wX = std::max(wStart, wMid);

  if (wEnd > wX) {
    return wX;
  } else if (wX == wStart) {
    return std::max(wMid, wEnd);
  } else if (wX == wMid) {
    return std::max(wStart, wEnd);
  }

  return wEnd;
}

// custom quicksort that partially sorts bbox data alongside the hilbert values
template <typename ArrayType>
void Flatbush<ArrayType>::sort(std::vector<uint32_t>& iValues,
                               size_t iLeft,
                               size_t iRight) noexcept {
  const auto wNodeSize = nodeSize();
  std::vector<std::size_t> wStack;
  wStack.reserve(iRight - iLeft);
  wStack.push_back(iLeft);
  wStack.push_back(iRight);

  while (wStack.size() > 1) {
    const auto wRight = wStack.back();
    wStack.pop_back();
    const auto wLeft = wStack.back();
    wStack.pop_back();

    if ((wRight - wLeft) > wNodeSize || wLeft < wRight) {
      const auto wPivot = medianOfThree(iValues, wLeft, wRight);
      auto wPivotLeft = wLeft - 1UL;
      auto wPivotRight = wRight + 1UL;

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

      wStack.push_back(wLeft);
      wStack.push_back(wPivotRight);
      wStack.push_back(wPivotRight + 1UL);
      wStack.push_back(wRight);
    }
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
  std::vector<size_t> wQueue;
  wQueue.reserve(wNodeSize << 2U);
  std::vector<size_t> wResults;
  wResults.reserve(detail::approximateResultsSize(mBounds, iBounds, wNumItems));

  while (wCanLoop) {
    // find the end index of the node
    const size_t wEnd = std::min(wNodeIndex + wNodeSize, upperBound(wNodeIndex));

    // search through child nodes
    for (size_t wPosition = wNodeIndex; wPosition < wEnd; ++wPosition) {
      // check if node bbox intersects with query bbox
      if (!detail::boxesIntersect(iBounds, mBoxes[wPosition])) {
        continue;
      }

      const size_t wIndex = mIsWideIndex ? mIndicesUint32[wPosition] : mIndicesUint16[wPosition];

      if (wNodeIndex >= wNumItems) {
        wQueue.push_back(wIndex);  // node; add it to the search queue
      } else if (!iFilterFn || iFilterFn(wIndex, mBoxes[wPosition])) {
        wResults.push_back(wIndex);  // leaf item
      }
    }

    if (wQueue.empty()) {
      break;
    }

    wNodeIndex = wQueue.back() >> 2U;  // for binary compatibility with JS
    wQueue.pop_back();
  }

  return wResults;
}

template <typename ArrayType>
std::vector<size_t> Flatbush<ArrayType>::neighbors(const Point<ArrayType>& iPoint,
                                                   size_t iMaxResults,
                                                   double iMaxDistance,
                                                   const FilterCb& iFilterFn) const noexcept {
  const auto wMaxDistSquared = iMaxDistance * iMaxDistance;
  const auto wCanLoop = canDoNeighbors(iPoint, iMaxResults, iMaxDistance, wMaxDistSquared);
  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();
  auto wNodeIndex = mBoxes.size() - 1UL;
  std::priority_queue<IndexDistance> wQueue;
  std::vector<size_t> wResults;
  wResults.reserve(std::min(wNumItems, iMaxResults));

  while (wCanLoop) {
    // find the end index of the node
    const auto wEnd = std::min(wNodeIndex + wNodeSize, upperBound(wNodeIndex));

    // search through child nodes
    for (auto wPosition = wNodeIndex; wPosition < wEnd; ++wPosition) {
      const size_t wIndex = (mIsWideIndex ? mIndicesUint32[wPosition] : mIndicesUint16[wPosition]);
      const auto wDistSquared = detail::computeDistanceSquared(iPoint, mBoxes[wPosition]);

      if (wDistSquared > wMaxDistSquared) {
        continue;
      } else if (wNodeIndex >= wNumItems) {
        wQueue.emplace(wIndex << 1U, wDistSquared);
      } else if (!iFilterFn || iFilterFn(wIndex, mBoxes[wPosition])) {
        // put an odd index if it's an item rather than a node, to recognize later
        wQueue.emplace((wIndex << 1U) + 1U, wDistSquared);  // leaf node
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

#endif  // FLATBUSH_FLATBUSH_H
