# Flatbush for C++

[![Ubuntu](https://github.com/chusitoo/flatbush/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/ubuntu.yml) [![Windows](https://github.com/chusitoo/flatbush/actions/workflows/windows.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/windows.yml) [![macOS](https://github.com/chusitoo/flatbush/actions/workflows/macos.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/macos.yml) [![CodeQL](https://github.com/chusitoo/flatbush/actions/workflows/codeql.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/codeql.yml)

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
    
#### CMake
```shell
mkdir build && cd build
cmake ..
make
ctest -C Release
``` 

#### Standalone
```shell
gcc test.cpp -lstdc++ -Wall -O2 -DFLATBUSH_SPAN -o test && ./test
```

```shell
clang++ -Wall -O2 -DFLATBUSH_SPAN -o test test.cpp && ./test
```

```shell
cl /EHsc /O2 /DFLATBUSH_SPAN test.cpp && .\test.exe
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
