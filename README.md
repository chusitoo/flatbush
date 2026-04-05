# Flatbush for C++

[![Format](https://github.com/chusitoo/flatbush/actions/workflows/format.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/format.yml)
[![CppCheck](https://github.com/chusitoo/flatbush/actions/workflows/cppcheck.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/cppcheck.yml)
[![CodeQL](https://github.com/chusitoo/flatbush/actions/workflows/codeql.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/codeql.yml)
[![Unit Tests](https://github.com/chusitoo/flatbush/actions/workflows/test.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/test.yml)
[![Fuzz tests](https://github.com/chusitoo/flatbush/actions/workflows/fuzz.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/fuzz.yml)
[![Benchmarks](https://github.com/chusitoo/flatbush/actions/workflows/bench.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/bench.yml)

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/chusitoo/flatbush)](https://github.com/chusitoo/flatbush/releases/latest)
[![Conan Center](https://img.shields.io/conan/v/flatbush)](https://conan.io/center/recipes/flatbush)
[![Vcpkg](https://img.shields.io/vcpkg/v/flatbush)](https://vcpkg.io/en/package/flatbush)

A C++ adaptation of the great Flatbush JS library which implements a packed Hilbert R-tree algorithm.

As such, unit tests and benchmarks are largely based on the original source in order to provide a close comparison.

## Acknowledgement

[A very fast static spatial index for 2D points and rectangles in JavaScript](https://github.com/mourner/flatbush) by Vladimir Agafonkin, licensed under [ISC](https://github.com/mourner/flatbush/blob/master/LICENSE)

## Usage

### Create and build the index

```cpp
using namespace flatbush;

// initialize the builder
FlatbushBuilder<double> builder;
// ... or preallocate buffer for 1000 items 
FlatbushBuilder<double> builder(1000);

// fill it with 1000 rectangles
for (const auto& box : boxes) {
    builder.add(box);
}

// perform the indexing
auto index = builder.finish();
```

### Searching a bounding box

```cpp
// make a bounding box query
Box<double> boundingBox{40, 40, 60, 60};
auto foundIds = index.search(boundingBox);

// make a bounding box query using a filter function 
auto filterEven = [](size_t id){ return id % 2 == 0; };
auto evenIds = index.search({40, 40, 60, 60}, filterEven);
```

### Searching for nearest neighbors

```cpp
// make a k-nearest-neighbors query
Point<double> targetPoint{40, 60};
auto neighborIds = index.neighbors(targetPoint);

// make a k-nearest-neighbors query with a maximum result threshold
auto neighborIds = index.neighbors({40, 60}, maxResults);

// make a k-nearest-neighbors query with a maximum result threshold and limit the distance 
auto neighborIds = index.neighbors(targetPoint, maxResults, maxDistance);

// make a k-nearest-neighbors query using a filter function
auto filterOdd = [](size_t id){ return id % 2 != 0; };
auto oddIds = index.neighbors({40, 60}, maxResults, maxDistance, filterOdd);
```

### Reconstruct from raw data

```cpp
// get the view to the raw array buffer
auto buffer = index.data();

// then pass the underlying data, specifying the template type
// NOTE: an exception will be thrown if template != encoded type
auto other = FlatbushBuilder<double>::from(buffer.data(), buffer.size());
// or, move the source vector into the builder
auto vector = std::vector<uint8_t>{buffer.begin(), buffer.end()};
auto other = FlatbushBuilder<double>::from(std::move(vector));
```

## Compiling
This is a single header library with the aim to support C++11 and up.

If the target compiler does not have support for C++20 features, namely the ```<span>``` header, a minimalistic implementation is available if **FLATBUSH_SPAN** flag is defined.

### SIMD Optimizations
The library automatically detects and uses SIMD instructions for improved performance. You can control the SIMD level with the following flags:

| ISA Level | GCC/Clang | MSVC |
|-----------|-----------|------|
| **SSE2** (default on x64) | `-msse2` | `/arch:SSE2` |
| **SSE3** | `-msse3` | N/A |
| **SSE4** | `-msse4`  | N/A |
| **AVX** | `-mavx` | `/arch:AVX` |
| **AVX2** | `-mavx2` | `/arch:AVX2` |
| **AVX512** | `-mavx512f -mavx512dq -mavx512vl` | `/arch:AVX512` |

### Unit tests
    
```shell
(cmake -B build/tests -DWITH_TESTS=ON && cmake --build build/tests -j $(nproc) && ./build/tests/unit_test)
```

### Bench tests

```shell
(cmake -B build/bench -DWITH_BENCHMARKS=ON && cmake --build build/bench -j $(nproc) && ./build/bench/bench_test)
```

### Fuzz tests

```shell
(cmake -B build/fuzz -DWITH_FUZZING=ON -DFUZZTEST_FUZZING_MODE=ON && cmake --build build/fuzz -j $(nproc) && ./build/fuzz/fuzz_test)
```

## Performance

On an i7-1185G7 @ 3.00GHz, Win11 version 25H2 / Ubuntu 24.04.2 LTS

bench test | clang 18.1.3  | gcc 13.3.0 | cl 19.29.30159
--- | --- | --- | ---
index 1000000 rectangles: | 85ms | 86ms | 106ms
1000 searches 10%: | 116ms | 113ms | 142ms
1000 searches 1%: | 20ms | 20ms | 21ms
1000 searches 0.01%: | 2ms | 2ms | 3ms
1000 searches of 100 neighbors: | 12ms | 12ms | 12ms
1 searches of 1000000 neighbors: | 80ms | 80ms | 63ms
100000 searches of 1 neighbors: | 188ms | 196ms | 207ms

Runner benchmarks over time for [gcc](https://chusitoo.github.io/flatbush/benchmarks/g++) and [clang](https://chusitoo.github.io/flatbush/benchmarks/clang++)
