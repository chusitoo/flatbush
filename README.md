# Flatbush for C++

[![CMake](https://github.com/chusitoo/flatbush/actions/workflows/cmake.yml/badge.svg)](https://github.com/chusitoo/flatbush/actions/workflows/cmake.yml)

A C++ implementation that mimics the great Flatbush JS library: https://github.com/mourner/flatbush

Unit tests and benchmarks are virtually identical in order to provide a close comparison to the original.

## Usage

### Create and build the index

```cpp
using namespace flatbush;

// initialize Flatbush for 1000 items
auto index = Flatbush<double>::create(1000);

// fill it with 1000 rectangles
for (auto p : items) {
    index.add(p.minX, p.minY, p.maxX, p.maxY);
}

// perform the indexing
index.finish();
```

### Searching a bounding box

```cpp
// make a bounding box query
auto foundIds = index.search(minX, minY, maxX, maxY);

// make a bounding box query using a filter function 
auto evenIds = index.search(minX, minY, maxX, maxY, [](size_t id){ return id % 2 == 0; });
```

### Searching for nearest neighbors

```cpp
// make a k-nearest-neighbors query
auto neighborIds = index.neighbors(x, y);

// make a k-nearest-neighbors query with a maximum result threshold
auto neighborIds = index.neighbors(x, y, maxResults);

// make a k-nearest-neighbors query with a maximum result threshold and limit the distance 
auto neighborIds = index.neighbors(x, y, maxResults, maxDistance);

// make a k-nearest-neighbors query using a filter function
auto evenIds = index.neighbors(x, y, maxResults, maxDistance, [](size_t id){ return id % 2 == 0; });
```

### Reconstruct from raw data
```cpp
// get the view to the raw array buffer
auto buffer = index.data();

// pass the underlying data as the only parameter
// NOTE: the template type has to match the type that was encoded 
auto other = Flatbush<double>::from(buffer.data());
// or
auto other = Flatbush<double>::from(&buffer[0]);
```

## Compiling
This is a single header library with the aim to support C++11 and up.

If the target compiler does not have support for C++20 features, namely the ```<span>``` header, a minimalistic implementation is available if **MINIMAL_SPAN** flag is defined.

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
gcc test.cpp -lstdc++ -Wall -O2 -DMINIMAL_SPAN -o test && ./test
```

```shell
clang++ -Wall -O2 -DMINIMAL_SPAN -o test test.cpp && ./test
```

```shell
cl /EHsc /O2 /DMINIMAL_SPAN test.cpp && .\test.exe
```

## Performance

On a i7-11850H @ 2.50GHz, Win10 version 20H2 / Ubuntu 20.04.3 LTS

bench test | clang 10.0.0 | gcc 9.3.0 | cl 14.29.30137.0
--- | --- | --- | ---
index 1000000 rectangles: | 75ms | 90ms | 105ms
1000 searches 10%: | 153ms | 134ms | 236ms
1000 searches 1%: | 29ms | 28ms | 33ms
1000 searches 0.01%: | 4ms | 5ms | 7ms
1000 searches of 100 neighbors: | 16ms | 16ms | 22ms
1 searches of 1000000 neighbors: | 85ms | 62ms | 57ms
100000 searches of 1 neighbors: | 365ms | 442ms | 587ms
```