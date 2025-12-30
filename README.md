# Flatbush for C++

[![Code format](https://github.com/chusitoo/flatbush/actions/workflows/format.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/format.yml)
[![Static code analysis](https://github.com/chusitoo/flatbush/actions/workflows/cppcheck.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/cppcheck.yml)
[![CodeQL](https://github.com/chusitoo/flatbush/actions/workflows/codeql.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/codeql.yml)
[![Unit Tests](https://github.com/chusitoo/flatbush/actions/workflows/test.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/test.yml)
[![Fuzz tests](https://github.com/chusitoo/flatbush/actions/workflows/fuzz.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/fuzz.yml)
[![Benchmarks](https://github.com/chusitoo/flatbush/actions/workflows/bench.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/bench.yml)

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/chusitoo/flatbush)](https://github.com/chusitoo/flatbush/releases/latest) [![Conan Center](https://img.shields.io/conan/v/flatbush)](https://conan.io/center/recipes/flatbush) [![Vcpkg](https://img.shields.io/vcpkg/v/flatbush)](https://vcpkg.io/en/package/flatbush)

A C++ adaptation of the great Flatbush JS library which implements a packed Hilbert R-tree algorithm.

As such, unit tests and benchmarks are virtually identical in order to provide a close comparison to the original.

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

// can reuse existing builder
// call builder.clear() or keep adding items
builder.add({ box.minX, box.minY, box.maxX, box.maxY });
auto other = builder.finish();
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

### Unit tests
    
```shell
(cmake -B build/tests -DWITH_TESTS=ON && cmake --build build/tests -j $(nproc) && ./build/tests/unit_test)
```

### Bench tests

```shell
(cmake -B build/bench -DWITH_BENCHMARKS=ON && cmake --build build/bench -j $(nproc) && ./build/bench/bench)
```

### Fuzz tests

```shell
(cmake -B build/fuzz -DWITH_FUZZING=ON -DFUZZTEST_FUZZING_MODE=ON && cmake --build build/fuzz -j $(nproc) && ./build/fuzz/fuzz)
```

## Performance

On a i7-11850H @ 2.50GHz, Win10 version 20H2 / Ubuntu 20.04.3 LTS

bench test | clang 10.0.0 | gcc 9.3.0 | cl 14.29.30137.0
--- | --- | --- | ---
index 1000000 rectangles: | 93ms | 112ms | 124ms
1000 searches 10%: | 120ms | 131ms | 194ms
1000 searches 1%: | 21ms | 23ms | 26ms
1000 searches 0.01%: | 3ms | 3ms | 4ms
1000 searches of 100 neighbors: | 12ms | 12ms | 17ms
1 searches of 1000000 neighbors: | 80ms | 59ms | 61ms
100000 searches of 1 neighbors: | 297ms | 363ms | 503ms
