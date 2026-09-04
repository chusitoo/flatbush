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
#ifndef FLATBUSH_SPAN
#include <span>         // for span
#endif
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
#elif defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)))
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
using std::span;
#else
template <typename Type>
class span {
  Type* mPtr = nullptr;
  size_t mLen = 0;

 public:
  span() noexcept = default;
  span(Type* iPtr, size_t iLen) noexcept : mPtr { iPtr }, mLen { iLen } {}
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
    return Box<OtherType> { static_cast<OtherType>(mMinX),
                            static_cast<OtherType>(mMinY),
                            static_cast<OtherType>(mMaxX),
                            static_cast<OtherType>(mMaxY) };
  }
};

template <typename ArrayType>
struct Point {
  ArrayType mX;
  ArrayType mY;

  template <typename OtherType>
  explicit operator Point<OtherType>() const noexcept {
    return Point<OtherType> { static_cast<OtherType>(mX), static_cast<OtherType>(mY) };
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

// A node is walked start to end, but it spans several cache lines and is reached by pointer
// chasing, so every line is requested up front rather than waiting for the stride detector
template <typename BoxType>
inline void prefetchNode(const BoxType* iBoxes, size_t iCount) noexcept {
  static constexpr size_t kCacheLine = 64;
  static constexpr size_t kBoxSize = sizeof(BoxType);
  static constexpr size_t kStride = (kCacheLine < kBoxSize) ? 1UL : kCacheLine / kBoxSize;

  for (size_t wIdx = 0; wIdx < iCount; wIdx += kStride) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(iBoxes + wIdx, 0, 3);
#elif defined(_MSC_VER) && defined(FLATBUSH_USE_SIMD)
    _mm_prefetch(bit_cast<const char*>(iBoxes + wIdx), _MM_HINT_T0);
#else
    (void)iBoxes;  // maybe unused
#endif
  }
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
    : std::integral_constant<bool, std::is_same<Type, Head>::value || is_contained<Type, Tail...>::value> {};

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, int8_t>::value, uint8_t>::type arrayTypeIndex() {
  return 0;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, uint8_t>::value, uint8_t>::type arrayTypeIndex() {
  return 1;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, int16_t>::value, uint8_t>::type arrayTypeIndex() {
  return 3;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, uint16_t>::value, uint8_t>::type arrayTypeIndex() {
  return 4;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, int32_t>::value, uint8_t>::type arrayTypeIndex() {
  return 5;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, uint32_t>::value, uint8_t>::type arrayTypeIndex() {
  return 6;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, float>::value, uint8_t>::type arrayTypeIndex() {
  return 7;
}

template <typename ArrayType>
constexpr typename std::enable_if<std::is_same<ArrayType, double>::value, uint8_t>::type arrayTypeIndex() {
  return 8;
}

template <typename ArrayType>
constexpr typename std::enable_if<
    !is_contained<ArrayType, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, float, double>::value,
    uint8_t>::type
arrayTypeIndex() {
  return gInvalidArrayType;
}

inline const char* arrayTypeName(size_t iIndex) {
  static constexpr auto kUnknownType = "unknown";
  static constexpr auto kArrayTypeNames = std::array<const char*, 9> { "int8_t",   "uint8_t",  "uint8_t",
                                                                       "int16_t",  "uint16_t", "int32_t",
                                                                       "uint32_t", "float",    "double" };
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

  // Calculate search area
  const auto wSearchWidth = wBoundsSearch.mMaxX - wBoundsSearch.mMinX;
  const auto wSearchHeight = wBoundsSearch.mMaxY - wBoundsSearch.mMinY;
  const auto wSearchArea = wSearchWidth * wSearchHeight;

  if (wIndexWidth <= 0 || wIndexHeight <= 0 || wIndexArea <= 0 || wSearchWidth <= 0 || wSearchHeight <= 0 ||
      !std::isfinite(wSearchWidth) || !std::isfinite(wSearchHeight) || !std::isfinite(wSearchArea) ||
      wSearchArea <= 0) {
    return 0UL;
  }

  // Approximate results size based as ratio of areas, assuming uniform distribution
  const auto wAreaRatio = wSearchArea / wIndexArea;
  if (!std::isfinite(wAreaRatio) || wAreaRatio > 1.0) {
    return iNumItems;
  }

  return static_cast<size_t>(static_cast<double>(iNumItems) * std::min(1.0, wAreaRatio * 1.5));
}

template <typename ArrayType>
inline bool boxesIntersect(const Box<ArrayType>& iQuery, const Box<ArrayType>& iBox) noexcept {
  // Bitwise or instead of logical or: the four comparisons are cheap and independent,
  // so evaluating them all beats short circuiting on unpredictable data
  return !((iQuery.mMaxX < iBox.mMinX) | (iQuery.mMaxY < iBox.mMinY) | (iQuery.mMinX > iBox.mMaxX) |
           (iQuery.mMinY > iBox.mMaxY));
}

// True when the query swallows the box whole, so every descendant of it matches
template <typename ArrayType>
inline bool boxContains(const Box<ArrayType>& iQuery, const Box<ArrayType>& iBox) noexcept {
  return !((iQuery.mMinX > iBox.mMinX) | (iQuery.mMinY > iBox.mMinY) | (iQuery.mMaxX < iBox.mMaxX) |
           (iQuery.mMaxY < iBox.mMaxY));
}

template <typename ArrayType>
inline void updateBounds(Box<ArrayType>& ioSrc, const Box<ArrayType>& iBox) noexcept {
  // Only float and double specialize below; hand-vectorising the integer types measured no
  // faster here, and slower for 8-bit boxes, which fit in a general purpose register anyway
  ioSrc.mMinX = std::min(ioSrc.mMinX, iBox.mMinX);
  ioSrc.mMinY = std::min(ioSrc.mMinY, iBox.mMinY);
  ioSrc.mMaxX = std::max(ioSrc.mMaxX, iBox.mMaxX);
  ioSrc.mMaxY = std::max(ioSrc.mMaxY, iBox.mMaxY);
}

template <typename ArrayType>
inline double axisDistance(ArrayType iValue, ArrayType iMin, ArrayType iMax) noexcept {
  const auto wValue = static_cast<double>(iValue);
  const auto wMin = static_cast<double>(iMin);
  const auto wMax = static_cast<double>(iMax);
  return std::max(0.0, std::max(wMin - wValue, wValue - wMax));
}

template <typename ArrayType>
inline double computeDistanceSquared(const Point<ArrayType>& iPoint, const Box<ArrayType>& iBox) noexcept {
  const auto wDistX = axisDistance(iPoint.mX, iBox.mMinX, iBox.mMaxX);
  const auto wDistY = axisDistance(iPoint.mY, iBox.mMinY, iBox.mMaxY);
  return wDistX * wDistX + wDistY * wDistY;
}

#if defined(FLATBUSH_USE_SIMD)
static constexpr auto kShuffleUnpackLo = _MM_SHUFFLE(1, 0, 1, 0);
static constexpr auto kShuffleUnpackHi = _MM_SHUFFLE(3, 2, 3, 2);
static constexpr auto kShuffleBroadcast1 = _MM_SHUFFLE(1, 1, 1, 1);
static constexpr auto kShuffleBlendMinMax = _MM_SHUFFLE(3, 2, 1, 0);
static constexpr auto kShuffleExchange01 = _MM_SHUFFLE2(0, 1);
static const auto kOffset32 = _mm_set1_epi32(std::numeric_limits<int32_t>::min());
static const auto kZeroPd = _mm_setzero_pd();
static const auto kZeroPs = _mm_setzero_ps();

static const auto kMaskAllOnes = _mm_set1_epi32(0xFFFF);
static const auto kMaskInterleave1 = _mm_set1_epi32(0x00FF00FF);
static const auto kMaskInterleave2 = _mm_set1_epi32(0x0F0F0F0F);
static const auto kMaskInterleave3 = _mm_set1_epi32(0x33333333);
static const auto kMaskInterleave4 = _mm_set1_epi32(0x55555555);

#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
static const auto kPermuteMinXY512 = _mm512_setr_epi64(0, 1, 4, 5, 8, 9, 12, 13);
static const auto kPermuteMaxXY512 = _mm512_setr_epi64(2, 3, 6, 7, 10, 11, 14, 15);
static const auto kPermuteXLoYHi = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
#endif

// True when no lane of a comparison mask is set
inline bool isNoneSet(__m128 iMask) noexcept {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  return _mm_testz_ps(iMask, iMask) != 0;
#else
  return _mm_movemask_ps(iMask) == 0;
#endif
}

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
  auto C = _mm_xor_si128(_mm_xor_si128(_mm_srli_epi32(c, 1), _mm_and_si128(b, _mm_srli_epi32(d, 1))), c);
  auto D = _mm_xor_si128(_mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(c, 1)), _mm_srli_epi32(d, 1)), d);

  a = A;
  b = B;
  c = C;
  d = D;
  A = _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(a, 2)), _mm_and_si128(b, _mm_srli_epi32(b, 2)));
  B = _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(b, 2)), _mm_and_si128(b, _mm_srli_epi32(_mm_xor_si128(a, b), 2)));
  C = _mm_xor_si128(C, _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(c, 2)), _mm_and_si128(b, _mm_srli_epi32(d, 2))));
  D = _mm_xor_si128(D,
                    _mm_xor_si128(_mm_and_si128(b, _mm_srli_epi32(c, 2)),
                                  _mm_and_si128(_mm_xor_si128(a, b), _mm_srli_epi32(d, 2))));

  a = A;
  b = B;
  c = C;
  d = D;
  A = _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(a, 4)), _mm_and_si128(b, _mm_srli_epi32(b, 4)));
  B = _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(b, 4)), _mm_and_si128(b, _mm_srli_epi32(_mm_xor_si128(a, b), 4)));
  C = _mm_xor_si128(C, _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(c, 4)), _mm_and_si128(b, _mm_srli_epi32(d, 4))));
  D = _mm_xor_si128(D,
                    _mm_xor_si128(_mm_and_si128(b, _mm_srli_epi32(c, 4)),
                                  _mm_and_si128(_mm_xor_si128(a, b), _mm_srli_epi32(d, 4))));

  // Final round
  a = A;
  b = B;
  c = C;
  d = D;
  C = _mm_xor_si128(C, _mm_xor_si128(_mm_and_si128(a, _mm_srli_epi32(c, 8)), _mm_and_si128(b, _mm_srli_epi32(d, 8))));
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
  const auto wMin = _mm_shuffle_ps(wQuery, wBox, kShuffleUnpackLo);
  const auto wMax = _mm_shuffle_ps(wBox, wQuery, kShuffleUnpackHi);
  return isNoneSet(_mm_cmplt_ps(wMax, wMin));
}

template <>
inline bool boxesIntersect<double>(const Box<double>& iQuery, const Box<double>& iBox) noexcept {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  const auto wQuery = _mm256_loadu_pd(&iQuery.mMinX);
  const auto wBox = _mm256_loadu_pd(&iBox.mMinX);
  const auto wMax = _mm256_permute2f128_pd(wQuery, wBox, 0x31);
  const auto wMin = _mm256_permute2f128_pd(wBox, wQuery, 0x20);
  return _mm256_movemask_pd(_mm256_cmp_pd(wMax, wMin, _CMP_LT_OQ)) == 0;
#else
  const auto wCmpMax = _mm_cmplt_pd(_mm_loadu_pd(&iQuery.mMaxX), _mm_loadu_pd(&iBox.mMinX));
  const auto wCmpMin = _mm_cmpgt_pd(_mm_loadu_pd(&iQuery.mMinX), _mm_loadu_pd(&iBox.mMaxX));
  return _mm_movemask_pd(_mm_or_pd(wCmpMax, wCmpMin)) == 0;
#endif
}

// Disjoint iff any lane of [bMaxX qMaxX bMaxY qMaxY] < [qMinX bMinX qMinY bMinY].
// Narrower integers reach this through the scalar template above: their whole box fits in a
// general purpose register, so the shuffling needed to vectorise costs more than it saves.
inline bool boxesIntersectEpi32(__m128i iQuery, __m128i iBox) noexcept {
  const auto wMin = _mm_unpacklo_epi32(iQuery, iBox);
  const auto wMax = _mm_unpackhi_epi32(iBox, iQuery);
  return isNoneSet(_mm_castsi128_ps(_mm_cmplt_epi32(wMax, wMin)));
}

template <>
inline bool boxesIntersect<int32_t>(const Box<int32_t>& iQuery, const Box<int32_t>& iBox) noexcept {
  return boxesIntersectEpi32(_mm_loadu_si128(bit_cast<const __m128i*>(&iQuery.mMinX)),
                             _mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX)));
}

template <>
inline bool boxesIntersect<uint32_t>(const Box<uint32_t>& iQuery, const Box<uint32_t>& iBox) noexcept {
  // Biasing into the signed domain turns the signed compare into an unsigned one
  return boxesIntersectEpi32(_mm_add_epi32(_mm_loadu_si128(bit_cast<const __m128i*>(&iQuery.mMinX)), kOffset32),
                             _mm_add_epi32(_mm_loadu_si128(bit_cast<const __m128i*>(&iBox.mMinX)), kOffset32));
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
  _mm_storeu_ps(&ioSrc.mMinX, _mm_shuffle_ps(wMins, wMaxs, kShuffleBlendMinMax));
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
inline double computeDistanceSquared<double>(const Point<double>& iPoint, const Box<double>& iBox) noexcept {
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
  const auto wDist = _mm_max_pd(kZeroPd, _mm_max_pd(_mm_sub_pd(wBoxMin, wPoint), _mm_sub_pd(wPoint, wBoxMax)));
  // Square and sum
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wResult = _mm_dp_pd(wDist, wDist, 0x31);
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE3
  const auto wDistSq = _mm_mul_pd(wDist, wDist);
  const auto wResult = _mm_hadd_pd(wDistSq, wDistSq);
#else
  const auto wDistSq = _mm_mul_pd(wDist, wDist);
  const auto wResult = _mm_add_pd(wDistSq, _mm_shuffle_pd(wDistSq, wDistSq, kShuffleExchange01));
#endif
  return _mm_cvtsd_f64(wResult);
}

template <>
inline double computeDistanceSquared<float>(const Point<float>& iPoint, const Box<float>& iBox) noexcept {
  const auto wBox = _mm_loadu_ps(&iBox.mMinX);
  const auto wPoint = _mm_castpd_ps(_mm_load_sd(bit_cast<const double*>(&iPoint.mX)));
  const auto wPoint2 = _mm_shuffle_ps(wPoint, wPoint, kShuffleUnpackLo);
  const auto wBoxMin = _mm_shuffle_ps(wBox, wBox, kShuffleUnpackLo);
  const auto wBoxMax = _mm_shuffle_ps(wBox, wBox, kShuffleUnpackHi);
  // Compute axis distances - using max to clamp to zero
  const auto wDist = _mm_max_ps(kZeroPs, _mm_max_ps(_mm_sub_ps(wBoxMin, wPoint2), _mm_sub_ps(wPoint2, wBoxMax)));
  // Square and sum
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wResult = _mm_dp_ps(wDist, wDist, 0x31);
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE3
  const auto wDistSq = _mm_mul_ps(wDist, wDist);
  const auto wResult = _mm_hadd_ps(wDistSq, wDistSq);
#else
  const auto wDistSq = _mm_mul_ps(wDist, wDist);
  const auto wResult = _mm_add_ps(wDistSq, _mm_shuffle_ps(wDistSq, wDistSq, kShuffleBroadcast1));
#endif
  return static_cast<double>(_mm_cvtss_f32(wResult));
}

template <>
inline double computeDistanceSquared<int8_t>(const Point<int8_t>& iPoint, const Box<int8_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}

template <>
inline double computeDistanceSquared<uint8_t>(const Point<uint8_t>& iPoint, const Box<uint8_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}

template <>
inline double computeDistanceSquared<int16_t>(const Point<int16_t>& iPoint, const Box<int16_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}

template <>
inline double computeDistanceSquared<uint16_t>(const Point<uint16_t>& iPoint, const Box<uint16_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}

template <>
inline double computeDistanceSquared<int32_t>(const Point<int32_t>& iPoint, const Box<int32_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}

template <>
inline double computeDistanceSquared<uint32_t>(const Point<uint32_t>& iPoint, const Box<uint32_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<float>>(iPoint), static_cast<Box<float>>(iBox));
}
#endif  // defined(FLATBUSH_USE_SIMD)

struct KeyIndex {
  uint32_t mKey;
  uint32_t mIndex;
};

// LSD radix over the whole 32-bit Hilbert key. Sorting the permutation keeps every pass at
// 8 bytes per item, and each of the 256 bucket cursors advances sequentially, so the scatter
// costs far less than the random placement a comparison sort's permutation would need.
inline void radixSortByKey(std::vector<KeyIndex>& ioPairs, std::vector<KeyIndex>& ioScratch) noexcept {
  static constexpr size_t kRadixBits = 8;
  static constexpr size_t kBuckets = size_t(1) << kRadixBits;
  static constexpr size_t kMask = kBuckets - 1;
  static constexpr size_t kKeyBits = 32;
  static constexpr size_t kPasses = kKeyBits / kRadixBits;
  static_assert(kPasses == 4, "The histogram below is unrolled for exactly four byte lanes");
  const size_t wCount = ioPairs.size();
  auto* wSrc = ioPairs.data();
  auto* wDst = ioScratch.data();
  size_t wOffsets[kPasses][kBuckets] = { { 0 } };

  // One read of the array feeds every pass, and spreading the counts across four tables keeps
  // consecutive increments off the same address
  for (size_t wIdx = 0; wIdx < wCount; ++wIdx) {
    const auto wKey = wSrc[wIdx].mKey;
    ++wOffsets[0][wKey & kMask];
    ++wOffsets[1][(wKey >> kRadixBits) & kMask];
    ++wOffsets[2][(wKey >> (2U * kRadixBits)) & kMask];
    ++wOffsets[3][(wKey >> (3U * kRadixBits)) & kMask];
  }

  for (size_t wPass = 0; wPass < kPasses; ++wPass) {
    size_t wRunning = 0;
    for (size_t wBucket = 0; wBucket < kBuckets; ++wBucket) {
      const auto wSize = wOffsets[wPass][wBucket];
      wOffsets[wPass][wBucket] = wRunning;
      wRunning += wSize;
    }

    const auto wShift = wPass * kRadixBits;
    for (size_t wIdx = 0; wIdx < wCount; ++wIdx) {
      wDst[wOffsets[wPass][(wSrc[wIdx].mKey >> wShift) & kMask]++] = wSrc[wIdx];
    }

    std::swap(wSrc, wDst);
  }
  // kPasses is even, so the sorted result lands back in ioPairs
}

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
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  const auto wHilbertWidth128 = _mm_broadcast_ss(&wHilbertWidth);
  const auto wHilbertHeight128 = _mm_broadcast_ss(&wHilbertHeight);
  const auto wDoubleMinX128 = _mm_broadcast_ss(&wDoubleMinX);
  const auto wDoubleMinY128 = _mm_broadcast_ss(&wDoubleMinY);
#else
  const auto wHilbertWidth128 = _mm_set1_ps(wHilbertWidth);
  const auto wHilbertHeight128 = _mm_set1_ps(wHilbertHeight);
  const auto wDoubleMinX128 = _mm_set1_ps(wDoubleMinX);
  const auto wDoubleMinY128 = _mm_set1_ps(wDoubleMinY);
#endif

  static const auto sumAxis = [](ArrayType iMin, ArrayType iMax) {
    return static_cast<float>(iMin) + static_cast<float>(iMax);
  };

  // Widening each corner one at a time keeps a single code path for every array type; the
  // Hilbert transform below is ~60 vector ops and dwarfs the cost of the gather
  for (; wIdx + 3 < iNumItems; wIdx += 4) {
    const auto wSumX = _mm_setr_ps(sumAxis(iBoxes[wIdx].mMinX, iBoxes[wIdx].mMaxX),
                                   sumAxis(iBoxes[wIdx + 1].mMinX, iBoxes[wIdx + 1].mMaxX),
                                   sumAxis(iBoxes[wIdx + 2].mMinX, iBoxes[wIdx + 2].mMaxX),
                                   sumAxis(iBoxes[wIdx + 3].mMinX, iBoxes[wIdx + 3].mMaxX));
    const auto wSumY = _mm_setr_ps(sumAxis(iBoxes[wIdx].mMinY, iBoxes[wIdx].mMaxY),
                                   sumAxis(iBoxes[wIdx + 1].mMinY, iBoxes[wIdx + 1].mMaxY),
                                   sumAxis(iBoxes[wIdx + 2].mMinY, iBoxes[wIdx + 2].mMaxY),
                                   sumAxis(iBoxes[wIdx + 3].mMinY, iBoxes[wIdx + 3].mMaxY));
    const auto wResultX = _mm_mul_ps(wHilbertWidth128, _mm_sub_ps(wSumX, wDoubleMinX128));
    const auto wResultY = _mm_mul_ps(wHilbertHeight128, _mm_sub_ps(wSumY, wDoubleMinY128));
    _mm_storeu_si128(bit_cast<__m128i*>(&wHilbertValues[wIdx]),
                     HilbertXYToIndex(_mm_cvtps_epi32(wResultX), _mm_cvtps_epi32(wResultY)));
  }
#endif  // defined(FLATBUSH_USE_SIMD)

  for (; wIdx < iNumItems; ++wIdx) {
    const auto& wBox = static_cast<Box<float>>(iBoxes[wIdx]);
    wHilbertValues[wIdx] = HilbertXYToIndex(static_cast<uint32_t>(wHilbertWidth *
                                                                  (wBox.mMinX + wBox.mMaxX - wDoubleMinX)),
                                            static_cast<uint32_t>(wHilbertHeight *
                                                                  (wBox.mMinY + wBox.mMaxY - wDoubleMinY)));
  }

  return wHilbertValues;
}

template <>
std::vector<uint32_t> computeHilbertValues<double>(size_t iNumItems,
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
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
  const auto wHilbertWidth512 = _mm512_set1_pd(wHilbertWidth);
  const auto wHilbertHeight512 = _mm512_set1_pd(wHilbertHeight);
  const auto wDoubleMinX512 = _mm512_set1_pd(wDoubleMinX);
  const auto wDoubleMinY512 = _mm512_set1_pd(wDoubleMinY);
  const auto wWidthHeight512 = _mm512_mask_blend_pd(0xAA, wHilbertWidth512, wHilbertHeight512);
  const auto wDoubleMinXY512 = _mm512_mask_blend_pd(0xAA, wDoubleMinX512, wDoubleMinY512);

  for (; wIdx + 3 < iNumItems; wIdx += 4) {
    const auto wBoxes01 = _mm512_loadu_pd(&iBoxes[wIdx].mMinX);
    const auto wBoxes23 = _mm512_loadu_pd(&iBoxes[wIdx + 2].mMinX);
    const auto wMin = _mm512_permutex2var_pd(wBoxes01, kPermuteMinXY512, wBoxes23);
    const auto wMax = _mm512_permutex2var_pd(wBoxes01, kPermuteMaxXY512, wBoxes23);
    const auto wResult = _mm256_permutevar8x32_epi32(_mm512_cvtpd_epi32(
                                                         _mm512_mul_pd(wWidthHeight512,
                                                                       _mm512_sub_pd(_mm512_add_pd(wMin, wMax),
                                                                                     wDoubleMinXY512))),
                                                     kPermuteXLoYHi);
    const auto wResultX = _mm256_castsi256_si128(wResult);
    const auto wResultY = _mm256_extracti32x4_epi32(wResult, 1);

    _mm_storeu_si128(bit_cast<__m128i*>(&wHilbertValues[wIdx]), HilbertXYToIndex(wResultX, wResultY));
  }
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  const auto wHilbertWidth256 = _mm256_broadcast_sd(&wHilbertWidth);
  const auto wHilbertHeight256 = _mm256_broadcast_sd(&wHilbertHeight);
  const auto wDoubleMinX256 = _mm256_broadcast_sd(&wDoubleMinX);
  const auto wDoubleMinY256 = _mm256_broadcast_sd(&wDoubleMinY);

  for (; wIdx + 3 < iNumItems; wIdx += 4) {
    const auto wBox0 = _mm256_loadu_pd(&iBoxes[wIdx].mMinX);
    const auto wBox1 = _mm256_loadu_pd(&iBoxes[wIdx + 1].mMinX);
    const auto wBox2 = _mm256_loadu_pd(&iBoxes[wIdx + 2].mMinX);
    const auto wBox3 = _mm256_loadu_pd(&iBoxes[wIdx + 3].mMinX);
    const auto wBoxes01Lo = _mm256_shuffle_pd(wBox0, wBox1, 0x0);
    const auto wBoxes01Hi = _mm256_shuffle_pd(wBox0, wBox1, 0xF);
    const auto wBoxes23Lo = _mm256_shuffle_pd(wBox2, wBox3, 0x0);
    const auto wBoxes23Hi = _mm256_shuffle_pd(wBox2, wBox3, 0xF);
    const auto wMinX = _mm256_permute2f128_pd(wBoxes01Lo, wBoxes23Lo, 0x20);
    const auto wMinY = _mm256_permute2f128_pd(wBoxes01Hi, wBoxes23Hi, 0x20);
    const auto wMaxX = _mm256_permute2f128_pd(wBoxes01Lo, wBoxes23Lo, 0x31);
    const auto wMaxY = _mm256_permute2f128_pd(wBoxes01Hi, wBoxes23Hi, 0x31);
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

    const auto wMinX = _mm_shuffle_pd(wBox0Lo, wBox1Lo, 0x0);
    const auto wMinY = _mm_shuffle_pd(wBox0Lo, wBox1Lo, 0x3);
    const auto wMaxX = _mm_shuffle_pd(wBox0Hi, wBox1Hi, 0x0);
    const auto wMaxY = _mm_shuffle_pd(wBox0Hi, wBox1Hi, 0x3);

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
    wHilbertValues.at(wIdx) = HilbertXYToIndex(uint32_t(wHilbertWidth * (wBox.mMinX + wBox.mMaxX - wDoubleMinX)),
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
  explicit FlatbushBuilder(size_t iNumItems = 10, uint16_t iNodeSize = gDefaultNodeSize) : mNodeSize(iNodeSize) {
    static_assert(detail::arrayTypeIndex<ArrayType>() != gInvalidArrayType,
                  "Unexpected typed array class. Expecting non 64-bit integral "
                  "or floating point.");

    mItems.reserve(iNumItems);
  }

  inline void clear() noexcept { mItems.clear(); }

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

  // Zero-copy: the bytes stay owned by the caller, who must outlive the index
  static Flatbush<ArrayType> fromView(span<const uint8_t> iBytes);

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

  // validate() rejects a null buffer, so the offset below is only ever taken on a live pointer
  // cppcheck-suppress nullPointerArithmetic
  return Flatbush<ArrayType>(std::vector<uint8_t>(iData, iData + iSize));
}

template <typename ArrayType>
Flatbush<ArrayType> FlatbushBuilder<ArrayType>::from(std::vector<uint8_t>&& iData) {
  validate(iData.data(), iData.size());

  return Flatbush<ArrayType>(std::move(iData));
}

template <typename ArrayType>
Flatbush<ArrayType> FlatbushBuilder<ArrayType>::fromView(span<const uint8_t> iBytes) {
  // Unlike the owning overloads, external bytes carry no alignment guarantee, so this has to
  // clear before validate reads the header through it
  static constexpr auto kAlignment = alignof(Box<ArrayType>) > alignof(uint32_t) ? alignof(Box<ArrayType>)
                                                                                 : alignof(uint32_t);

  if ((detail::bit_cast<uintptr_t>(iBytes.data()) + gHeaderByteSize) % kAlignment != 0UL) {
    throw std::invalid_argument("Data buffer must be aligned to " + std::to_string(kAlignment) + " bytes.");
  }

  validate(iBytes.data(), iBytes.size());

  return Flatbush<ArrayType>(iBytes);
}

template <typename ArrayType>
void FlatbushBuilder<ArrayType>::validate(const uint8_t* iData, size_t iSize) {
  static_assert(detail::arrayTypeIndex<ArrayType>() != gInvalidArrayType,
                "Unexpected typed array class. Expecting non 64-bit integral "
                "or floating point.");

  if (iSize < gHeaderByteSize) {
    throw std::invalid_argument("Data buffer size must be at least " + std::to_string(gHeaderByteSize) + " bytes.");
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
    throw std::invalid_argument("Got v" + std::to_string(wEncodedVersion) + " data when expected v" +
                                std::to_string(gVersion) + ".");
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

  const auto wNumItems = *detail::bit_cast<const uint32_t*>(&iData[4]);
  const auto wSize = Flatbush<ArrayType>::calculateDataSize(wNumItems, wNodeSize);
  if (wSize != iSize) {
    throw std::invalid_argument("Num items dictates a total size of " + std::to_string(wSize) +
                                ", but got buffer size " + std::to_string(iSize) + ".");
  }
}

template <typename ArrayType>
class Flatbush {
  using FilterCb = std::function<bool(size_t, const Box<ArrayType>&)>;
  // Must return a lower bound of the distance from the point to any point of the box,
  // otherwise the traversal prunes and orders on a value it cannot trust
  using DistanceCb = std::function<double(const Point<ArrayType>&, const Box<ArrayType>&)>;

 public:
  Flatbush(const Flatbush&) = delete;
  Flatbush& operator=(const Flatbush&) = delete;
  Flatbush(Flatbush&&) noexcept = default;
  Flatbush& operator=(Flatbush&&) noexcept = default;
  ~Flatbush() = default;

  std::vector<size_t> search(const Box<ArrayType>& iBounds, const FilterCb& iFilterFn = nullptr) const noexcept;

  // Without a distance callback, iMaxDistance is a Euclidean distance in index units; with
  // one, it is compared as-is against whatever that callback returns
  std::vector<size_t> neighbors(const Point<ArrayType>& iPoint,
                                size_t iMaxResults = gMaxResults,
                                double iMaxDistance = gMaxDistance,
                                const FilterCb& iFilterFn = nullptr,
                                const DistanceCb& iDistanceFn = nullptr) const noexcept;

  inline size_t nodeSize() const noexcept { return *detail::bit_cast<const uint16_t*>(mBytes.data() + 2); }

  inline size_t numItems() const noexcept { return *detail::bit_cast<const uint32_t*>(mBytes.data() + 4); }

  inline size_t indexSize() const noexcept { return mBoxes.size(); }

  inline bool isView() const noexcept { return mData.empty(); }

  inline span<const uint8_t> data() const noexcept { return mBytes; }

  friend class FlatbushBuilder<ArrayType>;

 private:
  static constexpr ArrayType kMaxValue = std::numeric_limits<ArrayType>::max();
  static constexpr ArrayType kMinValue = std::numeric_limits<ArrayType>::lowest();
  static constexpr auto kIsPacked = true;

  inline bool canDoSearch(const Box<ArrayType>& iBounds) const {
#if defined(_WIN32) || defined(_WIN64)
    // On Windows, isnan throws on anything that is not float, double or long double
    const auto wIsNanBounds = (std::isnan(static_cast<double>(iBounds.mMinX)) ||
                               std::isnan(static_cast<double>(iBounds.mMinY)) ||
                               std::isnan(static_cast<double>(iBounds.mMaxX)) ||
                               std::isnan(static_cast<double>(iBounds.mMaxY)));
#else
    const auto wIsNanBounds = (std::isnan(iBounds.mMinX) || std::isnan(iBounds.mMinY) || std::isnan(iBounds.mMaxX) ||
                               std::isnan(iBounds.mMaxY));
#endif

    return !wIsNanBounds && iBounds.mMaxX >= mBounds.mMinX && iBounds.mMinX <= mBounds.mMaxX &&
           iBounds.mMaxY >= mBounds.mMinY && iBounds.mMinY <= mBounds.mMaxY;
  }

  inline bool canDoNeighbors(const Point<ArrayType>& iPoint,
                             size_t iMaxResults,
                             double iMaxDistance,
                             double iThreshold,
                             const DistanceCb& iDistanceFn) const {
#if defined(_WIN32) || defined(_WIN64)
    // On Windows, isnan throws on anything that is not float, double or long double
    const auto wIsNanPoint = (std::isnan(static_cast<double>(iPoint.mX)) || std::isnan(static_cast<double>(iPoint.mY)));
#else
    const auto wIsNanPoint = (std::isnan(iPoint.mX) || std::isnan(iPoint.mY));
#endif

    const auto wDistance = iDistanceFn(iPoint, mBounds);

    return !wIsNanPoint && iMaxResults != 0UL && iMaxDistance > 0.0 && !std::isnan(wDistance) &&
           std::isnormal(iThreshold) && wDistance <= iThreshold;
  }

  Flatbush(uint32_t iNumItems, uint16_t iNodeSize);
  explicit Flatbush(std::vector<uint8_t>&& iData) noexcept;
  explicit Flatbush(span<const uint8_t> iBytes) noexcept;

  static size_t calculateDataSize(uint32_t iNumItems, uint32_t iNodeSize) noexcept;

  void create(std::vector<Box<ArrayType>>&& iItems) noexcept;
  void init(bool iIsPacked) noexcept;

  inline size_t upperBound(size_t iNodeIndex) const noexcept;

  inline size_t getIndex(size_t iPosition) const noexcept {
    return mIsWideIndex ? static_cast<size_t>(mIndicesUint32[iPosition])
                        : static_cast<size_t>(mIndicesUint16[iPosition]);
  }

  inline void setIndex(size_t iPosition, size_t iValue) noexcept {
    if (mIsWideIndex) {
      mIndicesUint32[iPosition] = static_cast<uint32_t>(iValue);
    } else {
      mIndicesUint16[iPosition] = static_cast<uint16_t>(iValue);
    }
  }

  inline size_t levelOf(size_t iNodeIndex) const noexcept;

  void collectContained(size_t iNodeIndex,
                        size_t iEnd,
                        size_t iLevel,
                        std::vector<size_t>& oResults,
                        const FilterCb& iFilterFn) const noexcept;

  std::vector<size_t> searchImpl(const Box<ArrayType>& iBounds, const FilterCb& iFilterFn) const noexcept;

  template <bool UseHeap>
  std::vector<size_t> neighborsImpl(const Point<ArrayType>& iPoint,
                                    size_t iMaxResults,
                                    double iThreshold,
                                    const FilterCb& iFilterFn,
                                    const DistanceCb& iDistanceFn) const noexcept;

  struct IndexDistance {
    IndexDistance(size_t iId, double iDistance) noexcept : mId(iId), mDistance(iDistance) {}
    bool operator<(const IndexDistance& iOther) const { return iOther.mDistance < mDistance; }

    size_t mId;
    double mDistance;
  };

  std::vector<uint8_t> mData;  // backing store, empty when the packed bytes are managed externally
  span<const uint8_t> mBytes;
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
Flatbush<ArrayType>::Flatbush(uint32_t iNumItems, uint16_t iNodeSize) {
  iNodeSize = std::min(std::max(iNodeSize, gMinNodeSize), gMaxNodeSize);

  mData.resize(calculateDataSize(iNumItems, iNodeSize), 0U);
  mData[0] = gValidityFlag;
  mData[1] = (gVersion << 4U) + detail::arrayTypeIndex<ArrayType>();
  *detail::bit_cast<uint16_t*>(&mData[2]) = iNodeSize;
  *detail::bit_cast<uint32_t*>(&mData[4]) = iNumItems;
  mBytes = { mData.data(), mData.size() };

  init(!kIsPacked);
}

template <typename ArrayType>
Flatbush<ArrayType>::Flatbush(std::vector<uint8_t>&& iData) noexcept
    : mData(std::move(iData)), mBytes(mData.data(), mData.size()) {
  init(kIsPacked);
}

template <typename ArrayType>
Flatbush<ArrayType>::Flatbush(span<const uint8_t> iBytes) noexcept : mBytes(iBytes) {
  init(kIsPacked);
}

template <typename ArrayType>
void Flatbush<ArrayType>::init(bool iIsPacked) noexcept {
  // Const is shed only to bind the typed views; externally managed bytes are never written to
  const auto wBase = const_cast<uint8_t*>(mBytes.data());
  const auto wNumItems = *detail::bit_cast<const uint32_t*>(wBase + 4);
  const auto wNodeSize = *detail::bit_cast<const uint16_t*>(wBase + 2);

  mBounds = { kMaxValue, kMaxValue, kMinValue, kMinValue };

  // Calculate the total number of nodes in the R-tree to allocate space for
  // and the index of each tree level (used in search later)
  size_t wCount = wNumItems;
  size_t wNumNodes = wNumItems;
  mLevelBounds.push_back(wNumNodes);

  do {
    wCount = (wCount + wNodeSize - 1UL) / wNodeSize;
    wNumNodes += wCount;
    mLevelBounds.push_back(wNumNodes);
  } while (wCount > 1UL);

  mIsWideIndex = wNumNodes > gMaxNumNodes;

  const size_t wNodesByteSize = wNumNodes * sizeof(Box<ArrayType>);
  mBoxes = { detail::bit_cast<Box<ArrayType>*>(wBase + gHeaderByteSize), wNumNodes };
  mIndicesUint16 = { detail::bit_cast<uint16_t*>(wBase + gHeaderByteSize + wNodesByteSize), wNumNodes };
  mIndicesUint32 = { detail::bit_cast<uint32_t*>(wBase + gHeaderByteSize + wNodesByteSize), wNumNodes };

  // Already-packed bytes leave nothing to fill in, so the tree starts out complete
  if (iIsPacked && wNumNodes > 0UL) {
    mPosition = wNumNodes;
    mBounds = mBoxes[wNumNodes - 1UL];
  }
}

template <typename ArrayType>
size_t Flatbush<ArrayType>::calculateDataSize(uint32_t iNumItems, uint32_t iNodeSize) noexcept {
  size_t wCount = iNumItems;
  size_t wNumNodes = iNumItems;

  do {
    wCount = (wCount + iNodeSize - 1UL) / iNodeSize;
    wNumNodes += wCount;
  } while (wCount > 1UL);

  const size_t wIndicesByteSize = wNumNodes * ((wNumNodes > gMaxNumNodes) ? sizeof(uint32_t) : sizeof(uint16_t));
  const size_t wNodesByteSize = wNumNodes * sizeof(Box<ArrayType>);

  return gHeaderByteSize + wNodesByteSize + wIndicesByteSize;
}

template <typename ArrayType>
void Flatbush<ArrayType>::create(std::vector<Box<ArrayType>>&& iItems) noexcept {
  for (size_t wIdx = 0UL; wIdx < iItems.size(); ++wIdx) {
    detail::updateBounds(mBounds, iItems[wIdx]);
  }
  mPosition = iItems.size();

  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();

  if (wNumItems <= wNodeSize) {
    for (size_t wIdx = 0UL; wIdx < wNumItems; ++wIdx) {
      setIndex(wIdx, wIdx);
      mBoxes[wIdx] = iItems[wIdx];
    }
    mBoxes[mPosition++] = mBounds;
    return;
  }

  // map item centers into Hilbert coordinate space and calculate Hilbert values
  auto wItemView = span<Box<ArrayType>>(iItems.data(), iItems.size());
  auto wHilbertValues = detail::computeHilbertValues(wNumItems, mBounds, wItemView);

  // sort a permutation by Hilbert value rather than dragging the boxes through the sort
  std::vector<detail::KeyIndex> wPairs(wNumItems);
  for (size_t wIdx = 0UL; wIdx < wNumItems; ++wIdx) {
    wPairs[wIdx] = { wHilbertValues[wIdx], static_cast<uint32_t>(wIdx) };
  }
  std::vector<uint32_t>().swap(wHilbertValues);
  std::vector<detail::KeyIndex> wScratch(wNumItems);
  detail::radixSortByKey(wPairs, wScratch);

  for (size_t wIdx = 0UL; wIdx < wNumItems; ++wIdx) {
    setIndex(wIdx, wPairs[wIdx].mIndex);
    mBoxes[wIdx] = iItems[wPairs[wIdx].mIndex];
  }

  for (size_t wIdx = 0UL, wPosition = 0UL; wIdx < mLevelBounds.size() - 1UL; ++wIdx) {
    const auto wEnd = mLevelBounds[wIdx];

    // generate a parent node for each block of consecutive <nodeSize> nodes
    while (wPosition < wEnd) {
      const auto wNodeIndex = wPosition << 2U;  // for binary compatibility with JS
      auto wNodeBox = mBoxes[wPosition];

      // calculate bbox for the new node
      for (size_t wCount = 0UL; wCount < wNodeSize && wPosition < wEnd; ++wCount, ++wPosition) {
        detail::updateBounds(wNodeBox, mBoxes[wPosition]);
      }

      // add the new node to the tree data
      setIndex(mPosition, wNodeIndex);
      mBoxes[mPosition++] = wNodeBox;
    }
  }
}

template <typename ArrayType>
size_t Flatbush<ArrayType>::upperBound(size_t iNodeIndex) const noexcept {
  static constexpr auto kSmallInput = 64UL;
  decltype(mLevelBounds.cbegin()) wIt;

  if (mLevelBounds.size() < kSmallInput) {
    for (wIt = mLevelBounds.cbegin(); wIt != mLevelBounds.cend() && *wIt <= iNodeIndex; ++wIt);
  } else {
    wIt = std::upper_bound(mLevelBounds.cbegin(), mLevelBounds.cend(), iNodeIndex);
  }

  return (mLevelBounds.cend() == wIt) ? mLevelBounds.back() : *wIt;
}

template <typename ArrayType>
size_t Flatbush<ArrayType>::levelOf(size_t iNodeIndex) const noexcept {
  size_t wLevel = 0UL;

  while (wLevel + 1UL < mLevelBounds.size() && mLevelBounds[wLevel] <= iNodeIndex) {
    ++wLevel;
  }

  return wLevel;
}

// Packing the tree bottom-up leaves every leaf of a subtree in one contiguous run, so a
// subtree the query swallows whole collapses to a descent to its first leaf and a flat sweep
template <typename ArrayType>
void Flatbush<ArrayType>::collectContained(size_t iNodeIndex,
                                           size_t iEnd,
                                           size_t iLevel,
                                           std::vector<size_t>& oResults,
                                           const FilterCb& iFilterFn) const noexcept {
  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();
  auto wPosition = iNodeIndex;
  auto wCount = iEnd - iNodeIndex;

  for (auto wDepth = iLevel; wDepth > 0UL; --wDepth) {
    wPosition = getIndex(wPosition) >> 2U;
    wCount = (wCount > wNumItems / wNodeSize) ? wNumItems : wCount * wNodeSize;
  }

  const auto wEnd = std::min(wPosition + wCount, wNumItems);

  if (iFilterFn) {
    for (; wPosition < wEnd; ++wPosition) {
      const auto wIndex = getIndex(wPosition);

      if (iFilterFn(wIndex, mBoxes[wPosition])) {
        oResults.push_back(wIndex);
      }
    }
  } else {
    for (; wPosition < wEnd; ++wPosition) {
      oResults.push_back(getIndex(wPosition));
    }
  }
}

template <typename ArrayType>
std::vector<size_t> Flatbush<ArrayType>::searchImpl(const Box<ArrayType>& iBounds,
                                                    const FilterCb& iFilterFn) const noexcept {
  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();
  auto wNodeIndex = mBoxes.size() - 1UL;
  // Node offsets are stored pre-multiplied by four, so the low bit is free to carry the flag
  auto wContained = detail::boxContains(iBounds, mBounds);
  std::vector<size_t> wQueue;
  wQueue.reserve(wNodeSize << 2U);
  std::vector<size_t> wResults;
  wResults.reserve(detail::approximateResultsSize(mBounds, iBounds, wNumItems));

  while (true) {
    // Split node-vs-leaf: the check is invariant across all children of a node
    if (wNodeIndex >= wNumItems) {
      // Internal node: only here does the enclosing level have to be looked up
      const size_t wEnd = std::min(wNodeIndex + wNodeSize, upperBound(wNodeIndex));

      if (wContained) {
        collectContained(wNodeIndex, wEnd, levelOf(wNodeIndex), wResults, iFilterFn);
      } else {
        for (size_t wPosition = wNodeIndex; wPosition < wEnd; ++wPosition) {
          if (detail::boxesIntersect(iBounds, mBoxes[wPosition])) {
            wQueue.push_back(getIndex(wPosition) |
                             static_cast<size_t>(detail::boxContains(iBounds, mBoxes[wPosition])));
          }
        }
      }
    } else {
      // Leaf node: the enclosing level always ends at the item count
      const size_t wEnd = std::min(wNodeIndex + wNodeSize, wNumItems);

      if (iFilterFn) {
        for (size_t wPosition = wNodeIndex; wPosition < wEnd; ++wPosition) {
          if (!wContained && !detail::boxesIntersect(iBounds, mBoxes[wPosition])) {
            continue;
          }
          const auto wIndex = getIndex(wPosition);
          if (iFilterFn(wIndex, mBoxes[wPosition])) {
            wResults.push_back(wIndex);
          }
        }
      } else if (wContained) {
        for (size_t wPosition = wNodeIndex; wPosition < wEnd; ++wPosition) {
          wResults.push_back(getIndex(wPosition));
        }
      } else {
        for (size_t wPosition = wNodeIndex; wPosition < wEnd; ++wPosition) {
          if (detail::boxesIntersect(iBounds, mBoxes[wPosition])) {
            wResults.push_back(getIndex(wPosition));
          }
        }
      }
    }

    if (wQueue.empty()) {
      break;
    }

    wContained = (wQueue.back() & 1UL) != 0UL;
    wNodeIndex = wQueue.back() >> 2U;  // for binary compatibility with JS
    wQueue.pop_back();
    detail::prefetchNode(&mBoxes[wNodeIndex], std::min(wNodeSize, mBoxes.size() - wNodeIndex));

    // Whenever the node just popped is a leaf it pushes no children, so the new top is the
    // one after it; requesting it now gives the load a whole node of work to hide behind
    if (!wQueue.empty()) {
      const auto wNextIndex = wQueue.back() >> 2U;
      detail::prefetchNode(&mBoxes[wNextIndex], std::min(wNodeSize, mBoxes.size() - wNextIndex));
    }
  }

  return wResults;
}

template <typename ArrayType>
std::vector<size_t> Flatbush<ArrayType>::search(const Box<ArrayType>& iBounds,
                                                const FilterCb& iFilterFn) const noexcept {
  if (!canDoSearch(iBounds)) {
    return {};
  }

  return searchImpl(iBounds, iFilterFn);
}

template <typename ArrayType>
template <bool UseHeap>
std::vector<size_t> Flatbush<ArrayType>::neighborsImpl(const Point<ArrayType>& iPoint,
                                                       size_t iMaxResults,
                                                       double iThreshold,
                                                       const FilterCb& iFilterFn,
                                                       const DistanceCb& iDistanceFn) const noexcept {
  const auto wNumItems = numItems();
  const auto wNodeSize = nodeSize();
  auto wNodeIndex = mBoxes.size() - 1UL;
  // Wanting a single result makes the closest leaf seen so far a valid bound: nothing
  // farther away can displace it, so anything beyond it need not be queued at all
  const auto wTrackNearest = iMaxResults == 1UL;
  auto wBound = iThreshold;
  std::vector<IndexDistance> wQueue;
  wQueue.reserve(wNodeSize << 2U);
  std::vector<size_t> wResults;
  wResults.reserve(std::min(wNumItems, iMaxResults));

  while (true) {
    // find the end index of the node; leaves always end at the item count
    const auto wIsInternalNode = wNodeIndex >= wNumItems;
    const auto wLevelEnd = wIsInternalNode ? upperBound(wNodeIndex) : wNumItems;
    const auto wEnd = std::min(wNodeIndex + wNodeSize, wLevelEnd);
    const auto wQueueSize = wQueue.size();

    for (auto wPosition = wNodeIndex; wPosition < wEnd; ++wPosition) {
      const auto wDistance = iDistanceFn(iPoint, mBoxes[wPosition]);

      if (wDistance > wBound) {
        continue;
      }

      const auto wIndex = getIndex(wPosition);

      if (wIsInternalNode || !iFilterFn || iFilterFn(wIndex, mBoxes[wPosition])) {
        wQueue.emplace_back((wIndex << 1U) + !wIsInternalNode, wDistance);
        if (UseHeap) std::push_heap(wQueue.begin(), wQueue.end());
        if (wTrackNearest && !wIsInternalNode && wDistance < wBound) {
          wBound = wDistance;
        }
      }
    }

    if (UseHeap) {  // Heap strategy: push_heap after each insert, pop from front
      while (!wQueue.empty() && (wQueue.front().mId & 1U)) {
        wResults.push_back(wQueue.front().mId >> 1U);
        std::pop_heap(wQueue.begin(), wQueue.end());
        wQueue.pop_back();

        if (wResults.size() >= iMaxResults) {
          return wResults;
        }
      }

      std::pop_heap(wQueue.begin(), wQueue.end());
    } else {  // Sorted-vector strategy: batch insert, sort+merge, pop from back
      if (wQueue.size() > wQueueSize) {
        const auto wMid = wQueue.begin() + static_cast<ptrdiff_t>(wQueueSize);
        std::sort(wMid, wQueue.end());
        std::inplace_merge(wQueue.begin(), wMid, wQueue.end());
      }

      while (!wQueue.empty() && (wQueue.back().mId & 1U)) {
        wResults.push_back(wQueue.back().mId >> 1U);
        wQueue.pop_back();

        if (wResults.size() >= iMaxResults) {
          return wResults;
        }
      }
    }

    if (wQueue.empty()) {
      break;
    }

    wNodeIndex = wQueue.back().mId >> 3U;  // 1 undo indexing + 2 for binary compatibility with JS
    wQueue.pop_back();
    detail::prefetchNode(&mBoxes[wNodeIndex], std::min(wNodeSize, mBoxes.size() - wNodeIndex));
  }

  return wResults;
}

template <typename ArrayType>
std::vector<size_t> Flatbush<ArrayType>::neighbors(const Point<ArrayType>& iPoint,
                                                   size_t iMaxResults,
                                                   double iMaxDistance,
                                                   const FilterCb& iFilterFn,
                                                   const DistanceCb& iDistanceFn) const noexcept {
  static constexpr auto kMergeThreshold = 128UL;
  static constexpr auto kUseHeap = true;
  const auto wNeedHeap = iMaxResults > kMergeThreshold;

  static const auto wDefaultFn = [](const Point<ArrayType>& iQuery, const Box<ArrayType>& iBox) noexcept {
    return detail::computeDistanceSquared(iQuery, iBox);
  };

  // A custom metric owns its units, so its threshold is taken as given; the built-in one
  // compares squared distances to keep the square root out of the traversal
  const DistanceCb wDistanceFn = iDistanceFn ? iDistanceFn : DistanceCb(wDefaultFn);
  const auto wThreshold = iDistanceFn ? iMaxDistance : iMaxDistance * iMaxDistance;

  if (!canDoNeighbors(iPoint, iMaxResults, iMaxDistance, wThreshold, wDistanceFn)) {
    return {};
  }

  if (wNeedHeap) {
    return neighborsImpl<kUseHeap>(iPoint, iMaxResults, wThreshold, iFilterFn, wDistanceFn);
  }

  return neighborsImpl<!kUseHeap>(iPoint, iMaxResults, wThreshold, iFilterFn, wDistanceFn);
}
}  // namespace flatbush

#endif  // FLATBUSH_FLATBUSH_H
