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
#define FLATBUSH_USE_SSE3 3
#define FLATBUSH_USE_SSE2 2

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
namespace flatbush {
using std::span;
}
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

inline void Interleave(std::array<uint32_t, 2>& x) {
#if defined(FLATBUSH_USE_SIMD) && FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE2
  auto v = _mm_setr_epi32(static_cast<int32_t>(x[0]), static_cast<int32_t>(x[1]), 0, 0);

  v = _mm_or_si128(v, _mm_slli_epi32(v, 8));
  v = _mm_and_si128(v, _mm_set1_epi32(0x00FF00FF));
  v = _mm_or_si128(v, _mm_slli_epi32(v, 4));
  v = _mm_and_si128(v, _mm_set1_epi32(0x0F0F0F0F));
  v = _mm_or_si128(v, _mm_slli_epi32(v, 2));
  v = _mm_and_si128(v, _mm_set1_epi32(0x33333333));
  v = _mm_or_si128(v, _mm_slli_epi32(v, 1));
  v = _mm_and_si128(v, _mm_set1_epi32(0x55555555));

  _mm_storeu_si64(&x[0], v);
#else
  x[0] = (x[0] | (x[0] << 8U)) & 0x00FF00FF;
  x[0] = (x[0] | (x[0] << 4U)) & 0x0F0F0F0F;
  x[0] = (x[0] | (x[0] << 2U)) & 0x33333333;
  x[0] = (x[0] | (x[0] << 1U)) & 0x55555555;

  x[1] = (x[1] | (x[1] << 8U)) & 0x00FF00FF;
  x[1] = (x[1] | (x[1] << 4U)) & 0x0F0F0F0F;
  x[1] = (x[1] | (x[1] << 2U)) & 0x33333333;
  x[1] = (x[1] | (x[1] << 1U)) & 0x55555555;
#endif
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
  std::array<uint32_t, 2> i;
  i[0] = x ^ y;
  i[1] = b | (0xFFFF ^ (i[0] | a));
  Interleave(i);
  return (i[1] << 1U) | i[0];
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
  return iValue < iMin ? iMin - iValue : std::max<double>(iValue - iMax, 0.0);
}

template <typename ArrayType>
inline double computeDistanceSquared(const Point<ArrayType>& iPoint,
                                     const Box<ArrayType>& iBox) noexcept {
  const auto wDistX = axisDistance(iPoint.mX, iBox.mMinX, iBox.mMaxX);
  const auto wDistY = axisDistance(iPoint.mY, iBox.mMinY, iBox.mMaxY);
  return wDistX * wDistX + wDistY * wDistY;
}

#if defined(FLATBUSH_USE_SIMD)
template <>
inline bool boxesIntersect<float>(const Box<float>& iQuery, const Box<float>& iBox) noexcept {
  const auto wMax = _mm_set_ps(iBox.mMaxY, iBox.mMaxX, iQuery.mMaxY, iQuery.mMaxX);
  const auto wMin = _mm_set_ps(iQuery.mMinY, iQuery.mMinX, iBox.mMinY, iBox.mMinX);
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
  const auto wMax = _mm256_set_pd(iBox.mMaxY, iBox.mMaxX, iQuery.mMaxY, iQuery.mMaxX);
  const auto wMin = _mm256_set_pd(iQuery.mMinY, iQuery.mMinX, iBox.mMinY, iBox.mMinX);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
  return _mm256_cmp_pd_mask(wMax, wMin, _CMP_LT_OQ) == 0;
#else
  const auto wCmp = _mm256_cmp_pd(wMax, wMin, _CMP_LT_OQ);
  return _mm256_testz_pd(wCmp, wCmp);
#endif
#else  // if FLATBUSH_USE_SIMD < FLATBUSH_USE_AVX
  const auto wQueryMax = _mm_set_pd(iQuery.mMaxY, iQuery.mMaxX);
  const auto wBoxMin = _mm_set_pd(iBox.mMinY, iBox.mMinX);
  const auto wCmp1 = _mm_cmplt_pd(wQueryMax, wBoxMin);
  const auto wQueryMin = _mm_set_pd(iQuery.mMinY, iQuery.mMinX);
  const auto wBoxMax = _mm_set_pd(iBox.mMaxY, iBox.mMaxX);
  const auto wCmp2 = _mm_cmpgt_pd(wQueryMin, wBoxMax);
  const auto wCombined = _mm_or_pd(wCmp1, wCmp2);
  return _mm_movemask_pd(wCombined) == 0;
#endif
}

template <>
inline bool boxesIntersect<int8_t>(const Box<int8_t>& iQuery, const Box<int8_t>& iBox) noexcept {
  const auto wMax = _mm_setr_epi8(
      iQuery.mMaxX, iQuery.mMaxY, iBox.mMaxX, iBox.mMaxY, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  const auto wMin = _mm_setr_epi8(
      iBox.mMinX, iBox.mMinY, iQuery.mMinX, iQuery.mMinY, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  const auto wCmp = _mm_cmplt_epi8(wMax, wMin);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  return _mm_testz_si128(wCmp, wCmp);
#else
  return _mm_movemask_epi8(wCmp) == 0;
#endif
}

template <>
inline bool boxesIntersect<uint8_t>(const Box<uint8_t>& iQuery, const Box<uint8_t>& iBox) noexcept {
  const auto wBox1 = static_cast<Box<int8_t>>(iQuery);
  const auto wBox2 = static_cast<Box<int8_t>>(iBox);
  const auto wOffset = _mm_set1_epi8(std::numeric_limits<int8_t>::min());
  const auto wMax = _mm_setr_epi8(
      wBox1.mMaxX, wBox1.mMaxY, wBox2.mMaxX, wBox2.mMaxY, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  const auto wMin = _mm_setr_epi8(
      wBox2.mMinX, wBox2.mMinY, wBox1.mMinX, wBox1.mMinY, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  const auto wCmp = _mm_cmplt_epi8(_mm_add_epi8(wMax, wOffset), _mm_add_epi8(wMin, wOffset));
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  return _mm_testz_si128(wCmp, wCmp);
#else
  return _mm_movemask_epi8(wCmp) == 0;
#endif
}

template <>
inline bool boxesIntersect<int16_t>(const Box<int16_t>& iQuery, const Box<int16_t>& iBox) noexcept {
  const auto wMax = _mm_setr_epi16(iQuery.mMaxX, iQuery.mMaxY, iBox.mMaxX, iBox.mMaxY, 0, 0, 0, 0);
  const auto wMin = _mm_setr_epi16(iBox.mMinX, iBox.mMinY, iQuery.mMinX, iQuery.mMinY, 0, 0, 0, 0);
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
  const auto wBox1 = static_cast<Box<int16_t>>(iQuery);
  const auto wBox2 = static_cast<Box<int16_t>>(iBox);
  const auto wOffset = _mm_set1_epi16(std::numeric_limits<int16_t>::min());
  const auto wMax = _mm_setr_epi16(wBox1.mMaxX, wBox1.mMaxY, wBox2.mMaxX, wBox2.mMaxY, 0, 0, 0, 0);
  const auto wMin = _mm_setr_epi16(wBox2.mMinX, wBox2.mMinY, wBox1.mMinX, wBox1.mMinY, 0, 0, 0, 0);
  const auto wCmp = _mm_cmplt_epi16(_mm_add_epi16(wMax, wOffset), _mm_add_epi16(wMin, wOffset));
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  return _mm_testz_si128(wCmp, wCmp);
#else
  return _mm_movemask_epi8(wCmp) == 0;
#endif
}

template <>
inline bool boxesIntersect<int32_t>(const Box<int32_t>& iQuery, const Box<int32_t>& iBox) noexcept {
  const auto wMax = _mm_setr_epi32(iQuery.mMaxX, iQuery.mMaxY, iBox.mMaxX, iBox.mMaxY);
  const auto wMin = _mm_setr_epi32(iBox.mMinX, iBox.mMinY, iQuery.mMinX, iQuery.mMinY);
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
  const auto wBox1 = static_cast<Box<int32_t>>(iQuery);
  const auto wBox2 = static_cast<Box<int32_t>>(iBox);
  const auto wOffset = _mm_set1_epi32(std::numeric_limits<int32_t>::min());
  const auto wMax = _mm_setr_epi32(wBox1.mMaxX, wBox1.mMaxY, wBox2.mMaxX, wBox2.mMaxY);
  const auto wMin = _mm_setr_epi32(wBox2.mMinX, wBox2.mMinY, wBox1.mMinX, wBox1.mMinY);
  const auto wCmp = _mm_cmplt_epi32(_mm_add_epi32(wMax, wOffset), _mm_add_epi32(wMin, wOffset));
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
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
  _mm_storeu_ps(&ioSrc.mMinX, _mm_mask_blend_ps(0xC, wMins, wMaxs));
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  _mm_storeu_ps(&ioSrc.mMinX, _mm_blend_ps(wMins, wMaxs, 0xC));
#else
  _mm_storeu_ps(&ioSrc.mMinX, _mm_shuffle_ps(wMins, wMaxs, _MM_SHUFFLE(1, 0, 1, 0)));
#endif
}

template <>
inline void updateBounds<double>(Box<double>& ioSrc, const Box<double>& iBox) noexcept {
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  const auto wCur = _mm256_loadu_pd(&ioSrc.mMinX);
  const auto wNew = _mm256_loadu_pd(&iBox.mMinX);
  const auto wMins = _mm256_min_pd(wCur, wNew);
  const auto wMaxs = _mm256_max_pd(wCur, wNew);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
  _mm256_storeu_pd(&ioSrc.mMinX, _mm256_mask_blend_pd(0xC, wMins, wMaxs));
#else  // if FLATBUSH_USE_SIMD == FLATBUSH_USE_AVX
  _mm256_storeu_pd(&ioSrc.mMinX, _mm256_blend_pd(wMins, wMaxs, 0xC));
#endif
#else  // FLATBUSH_USE_SIMD < FLATBUSH_USE_AVX
  const auto wCurMin = _mm_loadu_pd(&ioSrc.mMinX);
  const auto wNewMin = _mm_loadu_pd(&iBox.mMinX);
  _mm_storeu_pd(&ioSrc.mMinX, _mm_min_pd(wCurMin, wNewMin));
  const auto wCurMax = _mm_loadu_pd(&ioSrc.mMaxX);
  const auto wNewMax = _mm_loadu_pd(&iBox.mMaxX);
  _mm_storeu_pd(&ioSrc.mMaxX, _mm_max_pd(wCurMax, wNewMax));
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
  const auto wOffset = _mm_set1_epi16(std::numeric_limits<int16_t>::min());
  const auto wMins = _mm_sub_epi16(
      _mm_min_epi16(_mm_add_epi16(wCur, wOffset), _mm_add_epi16(wNew, wOffset)), wOffset);
  const auto wMaxs = _mm_sub_epi16(
      _mm_max_epi16(_mm_add_epi16(wCur, wOffset), _mm_add_epi16(wNew, wOffset)), wOffset);
#endif
  _mm_storeu_si64(&ioSrc.mMinX, _mm_unpacklo_epi32(wMins, wMaxs));
}

template <>
inline void updateBounds<int32_t>(Box<int32_t>& ioSrc, const Box<int32_t>& iBox) noexcept {
  const auto wCurrent = _mm_loadu_si128(detail::bit_cast<const __m128i*>(&ioSrc.mMinX));
  const auto wNewVals = _mm_loadu_si128(detail::bit_cast<const __m128i*>(&iBox.mMinX));
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
  _mm_storeu_si128(detail::bit_cast<__m128i*>(&ioSrc.mMinX), _mm_unpacklo_epi64(wMins, wMaxs));
}

template <>
inline void updateBounds<uint32_t>(Box<uint32_t>& ioSrc, const Box<uint32_t>& iBox) noexcept {
  const auto wCur = _mm_loadu_si128(detail::bit_cast<const __m128i*>(&ioSrc.mMinX));
  const auto wNew = _mm_loadu_si128(detail::bit_cast<const __m128i*>(&iBox.mMinX));
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE4
  const auto wMins = _mm_min_epu32(wCur, wNew);
  const auto wMaxs = _mm_max_epu32(wCur, wNew);
#else
  const auto wOffset = _mm_set1_epi32(std::numeric_limits<int32_t>::min());
  const auto wCurOff = _mm_add_epi32(wCur, wOffset);
  const auto wNewOff = _mm_add_epi32(wNew, wOffset);
  const auto wCmpMin = _mm_cmplt_epi32(wCurOff, wNewOff);
  const auto wCmpMax = _mm_cmpgt_epi32(wCurOff, wNewOff);
  const auto wMins = _mm_sub_epi32(
      _mm_or_si128(_mm_and_si128(wCmpMin, wCurOff), _mm_andnot_si128(wCmpMin, wNewOff)), wOffset);
  const auto wMaxs = _mm_sub_epi32(
      _mm_or_si128(_mm_and_si128(wCmpMax, wCurOff), _mm_andnot_si128(wCmpMax, wNewOff)), wOffset);
#endif
  _mm_storeu_si128(detail::bit_cast<__m128i*>(&ioSrc.mMinX), _mm_unpacklo_epi64(wMins, wMaxs));
}

template <>
inline double computeDistanceSquared<double>(const Point<double>& iPoint,
                                             const Box<double>& iBox) noexcept {
  const auto wPoint = _mm_set_pd(iPoint.mY, iPoint.mX);
  const auto wBoxMin = _mm_set_pd(iBox.mMinY, iBox.mMinX);
  const auto wBoxMax = _mm_set_pd(iBox.mMaxY, iBox.mMaxX);
  // Compute axis distances - using max to clamp is more efficient than conditional blending
  const auto wDistMin = _mm_max_pd(_mm_sub_pd(wBoxMin, wPoint), _mm_setzero_pd());
  const auto wDistMax = _mm_max_pd(_mm_sub_pd(wPoint, wBoxMax), _mm_setzero_pd());
  // Combine distances (only one will be non-zero per axis)
  const auto wDist = _mm_add_pd(wDistMin, wDistMax);
  // Square and sum
  const auto wDistSq = _mm_mul_pd(wDist, wDist);
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE3
  const auto wSum = _mm_hadd_pd(wDistSq, wDistSq);
#else
  const auto wSum = _mm_add_pd(wDistSq, _mm_unpackhi_pd(wDistSq, wDistSq));
#endif
  return _mm_cvtsd_f64(wSum);
}

template <>
inline double computeDistanceSquared<float>(const Point<float>& iPoint,
                                            const Box<float>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<double>>(iPoint), static_cast<Box<double>>(iBox));
}

template <>
inline double computeDistanceSquared<int8_t>(const Point<int8_t>& iPoint,
                                             const Box<int8_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<double>>(iPoint), static_cast<Box<double>>(iBox));
}

template <>
inline double computeDistanceSquared<uint8_t>(const Point<uint8_t>& iPoint,
                                              const Box<uint8_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<double>>(iPoint), static_cast<Box<double>>(iBox));
}

template <>
inline double computeDistanceSquared<int16_t>(const Point<int16_t>& iPoint,
                                              const Box<int16_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<double>>(iPoint), static_cast<Box<double>>(iBox));
}

template <>
inline double computeDistanceSquared<uint16_t>(const Point<uint16_t>& iPoint,
                                               const Box<uint16_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<double>>(iPoint), static_cast<Box<double>>(iBox));
}

template <>
inline double computeDistanceSquared<int32_t>(const Point<int32_t>& iPoint,
                                              const Box<int32_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<double>>(iPoint), static_cast<Box<double>>(iBox));
}

template <>
inline double computeDistanceSquared<uint32_t>(const Point<uint32_t>& iPoint,
                                               const Box<uint32_t>& iBox) noexcept {
  return computeDistanceSquared(static_cast<Point<double>>(iPoint), static_cast<Box<double>>(iBox));
}
#endif  // defined(FLATBUSH_USE_SIMD)

template <class ArrayType>
std::vector<uint32_t> computeHilbertValues(size_t iNumItems,
                                           const Box<ArrayType>& iBounds,
                                           const Box<ArrayType>* const iBoxes) {
  static constexpr auto wHalfScale = 0.5f;
  const auto wMinX = static_cast<float>(iBounds.mMinX);
  const auto wMinY = static_cast<float>(iBounds.mMinY);
  const auto wHilbertWidth = static_cast<float>(gMaxHilbert / (iBounds.mMaxX - iBounds.mMinX));
  const auto wHilbertHeight = static_cast<float>(gMaxHilbert / (iBounds.mMaxY - iBounds.mMinY));

  auto wIdx = 0UL;
  auto wHilbertValues = std::vector<uint32_t>(iNumItems);

#if defined(FLATBUSH_USE_SIMD)
#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX512
  const auto wBoundsMin512 = _mm512_setr_ps(wMinX,
                                            wMinY,
                                            wMinX,
                                            wMinY,
                                            wMinX,
                                            wMinY,
                                            wMinX,
                                            wMinY,
                                            wMinX,
                                            wMinY,
                                            wMinX,
                                            wMinY,
                                            wMinX,
                                            wMinY,
                                            wMinX,
                                            wMinY);
  const auto wScale512 = _mm512_setr_ps(wHilbertWidth,
                                        wHilbertHeight,
                                        wHilbertWidth,
                                        wHilbertHeight,
                                        wHilbertWidth,
                                        wHilbertHeight,
                                        wHilbertWidth,
                                        wHilbertHeight,
                                        wHilbertWidth,
                                        wHilbertHeight,
                                        wHilbertWidth,
                                        wHilbertHeight,
                                        wHilbertWidth,
                                        wHilbertHeight,
                                        wHilbertWidth,
                                        wHilbertHeight);
  const auto wHalf512 = _mm512_set1_ps(wHalfScale);

  for (auto wIdx1 = wIdx + 1,
            wIdx2 = wIdx + 2,
            wIdx3 = wIdx + 3,
            wIdx4 = wIdx + 4,
            wIdx5 = wIdx + 5,
            wIdx6 = wIdx + 6,
            wIdx7 = wIdx + 7;
       wIdx7 < iNumItems;
       wIdx += 8,
            wIdx1 += 8,
            wIdx2 += 8,
            wIdx3 += 8,
            wIdx4 += 8,
            wIdx5 += 8,
            wIdx6 += 8,
            wIdx7 += 8) {
    const auto wBox0 = static_cast<Box<float>>(iBoxes[wIdx]);
    const auto wBox1 = static_cast<Box<float>>(iBoxes[wIdx1]);
    const auto wBox2 = static_cast<Box<float>>(iBoxes[wIdx2]);
    const auto wBox3 = static_cast<Box<float>>(iBoxes[wIdx3]);
    const auto wBox4 = static_cast<Box<float>>(iBoxes[wIdx4]);
    const auto wBox5 = static_cast<Box<float>>(iBoxes[wIdx5]);
    const auto wBox6 = static_cast<Box<float>>(iBoxes[wIdx6]);
    const auto wBox7 = static_cast<Box<float>>(iBoxes[wIdx7]);

    const auto wBoxMin = _mm512_setr_ps(wBox0.mMinX,
                                        wBox0.mMinY,
                                        wBox1.mMinX,
                                        wBox1.mMinY,
                                        wBox2.mMinX,
                                        wBox2.mMinY,
                                        wBox3.mMinX,
                                        wBox3.mMinY,
                                        wBox4.mMinX,
                                        wBox4.mMinY,
                                        wBox5.mMinX,
                                        wBox5.mMinY,
                                        wBox6.mMinX,
                                        wBox6.mMinY,
                                        wBox7.mMinX,
                                        wBox7.mMinY);
    const auto wBoxMax = _mm512_setr_ps(wBox0.mMaxX,
                                        wBox0.mMaxY,
                                        wBox1.mMaxX,
                                        wBox1.mMaxY,
                                        wBox2.mMaxX,
                                        wBox2.mMaxY,
                                        wBox3.mMaxX,
                                        wBox3.mMaxY,
                                        wBox4.mMaxX,
                                        wBox4.mMaxY,
                                        wBox5.mMaxX,
                                        wBox5.mMaxY,
                                        wBox6.mMaxX,
                                        wBox6.mMaxY,
                                        wBox7.mMaxX,
                                        wBox7.mMaxY);

    const auto wSum = _mm512_add_ps(wBoxMin, wBoxMax);
    const auto wCenter = _mm512_mul_ps(wSum, wHalf512);
    const auto wOffset = _mm512_sub_ps(wCenter, wBoundsMin512);
    const auto wResult = _mm512_mul_ps(wOffset, wScale512);

    std::array<uint32_t, 16> wResInt;
    _mm512_store_si512(detail::bit_cast<__m512i*>(wResInt.data()), _mm512_cvtps_epi32(wResult));
    wHilbertValues.at(wIdx) = detail::HilbertXYToIndex(wResInt[0], wResInt[1]);
    wHilbertValues.at(wIdx1) = detail::HilbertXYToIndex(wResInt[2], wResInt[3]);
    wHilbertValues.at(wIdx2) = detail::HilbertXYToIndex(wResInt[4], wResInt[5]);
    wHilbertValues.at(wIdx3) = detail::HilbertXYToIndex(wResInt[6], wResInt[7]);
    wHilbertValues.at(wIdx4) = detail::HilbertXYToIndex(wResInt[8], wResInt[9]);
    wHilbertValues.at(wIdx5) = detail::HilbertXYToIndex(wResInt[10], wResInt[11]);
    wHilbertValues.at(wIdx6) = detail::HilbertXYToIndex(wResInt[12], wResInt[13]);
    wHilbertValues.at(wIdx7) = detail::HilbertXYToIndex(wResInt[14], wResInt[15]);
  }
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX
  const auto wBoundsMin256 = _mm256_setr_ps(wMinX, wMinY, wMinX, wMinY, wMinX, wMinY, wMinX, wMinY);
  const auto wScale256 = _mm256_setr_ps(wHilbertWidth,
                                        wHilbertHeight,
                                        wHilbertWidth,
                                        wHilbertHeight,
                                        wHilbertWidth,
                                        wHilbertHeight,
                                        wHilbertWidth,
                                        wHilbertHeight);
  const auto wHalf256 = _mm256_set1_ps(wHalfScale);

  for (auto wIdx1 = wIdx + 1, wIdx2 = wIdx + 2, wIdx3 = wIdx + 3; wIdx3 < iNumItems;
       wIdx += 4, wIdx1 += 4, wIdx2 += 4, wIdx3 += 4) {
    const auto wBox0 = static_cast<Box<float>>(iBoxes[wIdx]);
    const auto wBox1 = static_cast<Box<float>>(iBoxes[wIdx1]);
    const auto wBox2 = static_cast<Box<float>>(iBoxes[wIdx2]);
    const auto wBox3 = static_cast<Box<float>>(iBoxes[wIdx3]);

    const auto wBoxMin = _mm256_setr_ps(wBox0.mMinX,
                                        wBox0.mMinY,
                                        wBox1.mMinX,
                                        wBox1.mMinY,
                                        wBox2.mMinX,
                                        wBox2.mMinY,
                                        wBox3.mMinX,
                                        wBox3.mMinY);
    const auto wBoxMax = _mm256_setr_ps(wBox0.mMaxX,
                                        wBox0.mMaxY,
                                        wBox1.mMaxX,
                                        wBox1.mMaxY,
                                        wBox2.mMaxX,
                                        wBox2.mMaxY,
                                        wBox3.mMaxX,
                                        wBox3.mMaxY);

    const auto wSum = _mm256_add_ps(wBoxMin, wBoxMax);
    const auto wCenter = _mm256_mul_ps(wSum, wHalf256);
    const auto wOffset = _mm256_sub_ps(wCenter, wBoundsMin256);
    const auto wResult = _mm256_mul_ps(wOffset, wScale256);

#if FLATBUSH_USE_SIMD >= FLATBUSH_USE_AVX2
    std::array<uint32_t, 8> wResInt;
    _mm256_store_si256(detail::bit_cast<__m256i*>(wResInt.data()), _mm256_cvtps_epi32(wResult));
    wHilbertValues.at(wIdx) = detail::HilbertXYToIndex(wResInt[0], wResInt[1]);
    wHilbertValues.at(wIdx1) = detail::HilbertXYToIndex(wResInt[2], wResInt[3]);
    wHilbertValues.at(wIdx2) = detail::HilbertXYToIndex(wResInt[4], wResInt[5]);
    wHilbertValues.at(wIdx3) = detail::HilbertXYToIndex(wResInt[6], wResInt[7]);
#else
    std::array<float, 8> wRes;
    _mm256_store_ps(wRes.data(), wResult);
    wHilbertValues.at(wIdx) =
        detail::HilbertXYToIndex(static_cast<uint32_t>(wRes[0]), static_cast<uint32_t>(wRes[1]));
    wHilbertValues.at(wIdx1) =
        detail::HilbertXYToIndex(static_cast<uint32_t>(wRes[2]), static_cast<uint32_t>(wRes[3]));
    wHilbertValues.at(wIdx2) =
        detail::HilbertXYToIndex(static_cast<uint32_t>(wRes[4]), static_cast<uint32_t>(wRes[5]));
    wHilbertValues.at(wIdx3) =
        detail::HilbertXYToIndex(static_cast<uint32_t>(wRes[6]), static_cast<uint32_t>(wRes[7]));
#endif
  }
#elif FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE2
  const auto wBoundsMin = _mm_setr_ps(wMinX, wMinY, wMinX, wMinY);
  const auto wScale = _mm_setr_ps(wHilbertWidth, wHilbertHeight, wHilbertWidth, wHilbertHeight);
  const auto wHalf = _mm_set1_ps(wHalfScale);

  for (auto wIdx1 = wIdx + 1; wIdx1 < iNumItems; wIdx += 2, wIdx1 += 2) {
    const auto wBox1 = static_cast<Box<float>>(iBoxes[wIdx]);
    const auto wBox2 = static_cast<Box<float>>(iBoxes[wIdx1]);
    const auto wBoxMin = _mm_setr_ps(wBox1.mMinX, wBox1.mMinY, wBox2.mMinX, wBox2.mMinY);
    const auto wBoxMax = _mm_setr_ps(wBox1.mMaxX, wBox1.mMaxY, wBox2.mMaxX, wBox2.mMaxY);

    const auto wSum = _mm_add_ps(wBoxMin, wBoxMax);
    const auto wCenter = _mm_mul_ps(wSum, wHalf);
    const auto wOffset = _mm_sub_ps(wCenter, wBoundsMin);
    const auto wResult = _mm_mul_ps(wOffset, wScale);

    std::array<float, 4> wRes;
    _mm_store_ps(wRes.data(), wResult);
    wHilbertValues.at(wIdx) =
        detail::HilbertXYToIndex(static_cast<uint32_t>(wRes[0]), static_cast<uint32_t>(wRes[1]));
    wHilbertValues.at(wIdx1) =
        detail::HilbertXYToIndex(static_cast<uint32_t>(wRes[2]), static_cast<uint32_t>(wRes[3]));
  }
#endif  // FLATBUSH_USE_SIMD >= FLATBUSH_USE_SSE2
#endif  // defined(FLATBUSH_USE_SIMD)

  for (; wIdx < iNumItems; ++wIdx) {
    const auto& wBox = static_cast<Box<float>>(iBoxes[wIdx]);
    wHilbertValues.at(wIdx) = detail::HilbertXYToIndex(
        uint32_t(wHilbertWidth * ((wBox.mMinX + wBox.mMaxX) * wHalfScale - wMinX)),
        uint32_t(wHilbertHeight * ((wBox.mMinY + wBox.mMaxY) * wHalfScale - wMinY)));
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

  Flatbush<ArrayType> finish() const;
  static Flatbush<ArrayType> from(const uint8_t* iData, size_t iSize);
  static Flatbush<ArrayType> from(std::vector<uint8_t>&& iData);

 private:
  static void validate(const uint8_t* iData, size_t iSize);
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

    const auto wDistX = axisDistance(iPoint.mX, mBounds.mMinX, mBounds.mMaxX);
    const auto wDistY = axisDistance(iPoint.mY, mBounds.mMinY, mBounds.mMaxY);
    const auto wDistance = wDistX * wDistX + wDistY * wDistY;

    return !wIsNanPoint && iMaxResults != 0UL && iMaxDistance > 0.0 && !std::isnan(wDistance) &&
           std::isnormal(iMaxDistSquared) && wDistance <= iMaxDistSquared;
  }

  Flatbush(uint32_t iNumItems, uint16_t iNodeSize) noexcept;
  Flatbush(const uint8_t* iData, size_t iSize) noexcept;
  explicit Flatbush(std::vector<uint8_t>&& iData) noexcept;

  size_t add(const Box<ArrayType>& iBox) noexcept;
  void finish() noexcept;
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

  mData.resize(mData.capacity(), 0U);
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
Flatbush<ArrayType>::Flatbush(std::vector<uint8_t>&& iData) noexcept {
  const auto wNodeSize = *detail::bit_cast<const uint16_t*>(&iData.at(2));
  const auto wNumItems = *detail::bit_cast<const uint32_t*>(&iData.at(4));
  init(wNumItems, wNodeSize);

  mData = std::move(iData);
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
  detail::updateBounds(mBounds, iBox);
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

  // map item centers into Hilbert coordinate space and calculate Hilbert values
  auto wHilbertValues = detail::computeHilbertValues(wNumItems, mBounds, mBoxes.data());
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
  // Calculate intersection area
  const auto wWidth =
      std::min(mBounds.mMaxX, iBounds.mMaxX) - std::max(mBounds.mMinX, iBounds.mMinX);
  const auto wHeight =
      std::min(mBounds.mMaxY, iBounds.mMaxY) - std::max(mBounds.mMinY, iBounds.mMinY);
  // Approximate results vector size based on intersection area, assuming uniform distribution
  const auto wSearchArea = (iBounds.mMaxX - iBounds.mMinX) * (iBounds.mMaxY - iBounds.mMinY);
  wResults.reserve(
      (wWidth > ArrayType{0} && wHeight > ArrayType{0} && wSearchArea > ArrayType{0})
          ? static_cast<size_t>(static_cast<double>(wNumItems) * wSearchArea / (wWidth * wHeight))
          : 0UL);

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
      const auto wDistance = detail::computeDistanceSquared(iPoint, mBoxes[wPosition]);

      if (wDistance > wMaxDistSquared) {
        continue;
      } else if (wNodeIndex >= wNumItems) {
        wQueue.emplace(wIndex << 1U, wDistance);
      } else if (!iFilterFn || iFilterFn(wIndex, mBoxes[wPosition])) {
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
