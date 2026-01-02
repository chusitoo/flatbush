/*
MIT License

Copyright (c) 2025 Alex Emirov

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

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "flatbush.h"

using randomDouble = std::uniform_real_distribution<double>;

struct BenchmarkData {
  std::vector<uint8_t> mIndexData;
  std::vector<double> mCoords;
  std::vector<double> mBoxes100;
  std::vector<double> mBoxes10;
  std::vector<double> mBoxes1;

  static BenchmarkData instance() {
    static BenchmarkData kInstance;
    return kInstance;
  }

  static void addRandomBox(std::vector<double>& iBoxes, double iBoxSize) {
    static std::mt19937 wEngine(5489031744);
    auto wMinX = randomDouble(0.0, 100.0 - iBoxSize)(wEngine);
    auto wMinY = randomDouble(0.0, 100.0 - iBoxSize)(wEngine);
    auto wMaxX = wMinX + randomDouble(0.0, iBoxSize)(wEngine);
    auto wMaxY = wMinY + randomDouble(0.0, iBoxSize)(wEngine);
    iBoxes.push_back(wMinX);
    iBoxes.push_back(wMinY);
    iBoxes.push_back(wMaxX);
    iBoxes.push_back(wMaxY);
  }

 private:
  BenchmarkData() {
    size_t wNumItems = 1000000;
    size_t wNumTests = 1000;
    size_t wNodeSize = 16;

    // Generate random coordinates
    for (size_t wCount = 0; wCount < wNumItems; ++wCount) {
      addRandomBox(mCoords, 1.0);
    }

    // Generate test boxes
    for (size_t wCount = 0; wCount < wNumTests; ++wCount) {
      addRandomBox(mBoxes100, 100.0 * std::sqrt(0.1));
      addRandomBox(mBoxes10, 10.0);
      addRandomBox(mBoxes1, 1.0);
    }

    // Build the index
    flatbush::FlatbushBuilder<double> wBuilder(wNodeSize);
    for (size_t wIdx = 0; wIdx < mCoords.size(); wIdx += 4) {
      wBuilder.add({mCoords[wIdx], mCoords[wIdx + 1], mCoords[wIdx + 2], mCoords[wIdx + 3]});
    }
    auto wIndex = wBuilder.finish();
    auto span = wIndex.data();
    mIndexData.assign(span.begin(), span.end());
  }
};

static BenchmarkData gData = BenchmarkData::instance();
static flatbush::Flatbush<double> gIndex =
    flatbush::FlatbushBuilder<double>::from(gData.mIndexData.data(), gData.mIndexData.size());

static void BM_Index1M(benchmark::State& state) {
  static constexpr auto kNumItems = 1000000;
  static constexpr auto kNodeSize = 16;
  std::vector<double> wCoords;

  for (size_t wCount = 0; wCount < kNumItems; ++wCount) {
    BenchmarkData::addRandomBox(wCoords, 1.0);
  }

  for (auto _ : state) {
    flatbush::FlatbushBuilder<double> wBuilder(kNodeSize);
    for (size_t wIdx = 0; wIdx < wCoords.size(); wIdx += 4) {
      wBuilder.add({wCoords[wIdx], wCoords[wIdx + 1], wCoords[wIdx + 2], wCoords[wIdx + 3]});
    }
    auto wIndex = wBuilder.finish();
    benchmark::DoNotOptimize(wIndex);
  }

  state.SetItemsProcessed(state.iterations() * kNumItems);
}
BENCHMARK(BM_Index1M);

static void BM_Search10Percent(benchmark::State& state) {
  for (auto _ : state) {
    for (size_t wIdx = 0; wIdx < gData.mBoxes100.size(); wIdx += 4) {
      auto result = gIndex.search({gData.mBoxes100[wIdx],
                                   gData.mBoxes100[wIdx + 1],
                                   gData.mBoxes100[wIdx + 2],
                                   gData.mBoxes100[wIdx + 3]});
      benchmark::DoNotOptimize(result);
    }
  }

  state.SetItemsProcessed(state.iterations() * (gData.mBoxes100.size() / 4));
}
BENCHMARK(BM_Search10Percent);

static void BM_Search1Percent(benchmark::State& state) {
  for (auto _ : state) {
    for (size_t wIdx = 0; wIdx < gData.mBoxes10.size(); wIdx += 4) {
      auto result = gIndex.search({gData.mBoxes10[wIdx],
                                   gData.mBoxes10[wIdx + 1],
                                   gData.mBoxes10[wIdx + 2],
                                   gData.mBoxes10[wIdx + 3]});
      benchmark::DoNotOptimize(result);
    }
  }

  state.SetItemsProcessed(state.iterations() * (gData.mBoxes10.size() / 4));
}
BENCHMARK(BM_Search1Percent);

static void BM_Search001Percent(benchmark::State& state) {
  for (auto _ : state) {
    for (size_t wIdx = 0; wIdx < gData.mBoxes1.size(); wIdx += 4) {
      auto result = gIndex.search({gData.mBoxes1[wIdx],
                                   gData.mBoxes1[wIdx + 1],
                                   gData.mBoxes1[wIdx + 2],
                                   gData.mBoxes1[wIdx + 3]});
      benchmark::DoNotOptimize(result);
    }
  }

  state.SetItemsProcessed(state.iterations() * (gData.mBoxes1.size() / 4));
}
BENCHMARK(BM_Search001Percent);

static void BM_Neighbors100(benchmark::State& state) {
  static constexpr auto kNumTests = 1000UL;

  for (auto _ : state) {
    for (size_t wIdx = 0; wIdx < kNumTests; ++wIdx) {
      auto result = gIndex.neighbors({gData.mCoords[4 * wIdx], gData.mCoords[4 * wIdx + 1]}, 100);
      benchmark::DoNotOptimize(result);
    }
  }

  state.SetItemsProcessed(state.iterations() * kNumTests);
}
BENCHMARK(BM_Neighbors100);

static void BM_NeighborsAll(benchmark::State& state) {
  for (auto _ : state) {
    auto result = gIndex.neighbors({gData.mCoords[0], gData.mCoords[1]}, gData.mCoords.size() / 4);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_NeighborsAll);

static void BM_Neighbors1(benchmark::State& state) {
  static constexpr auto kNumTests = 100000UL;

  for (auto _ : state) {
    for (size_t wIdx = 0; wIdx < kNumTests; ++wIdx) {
      auto result = gIndex.neighbors({gData.mCoords[4 * wIdx], gData.mCoords[4 * wIdx + 1]}, 1);
      benchmark::DoNotOptimize(result);
    }
  }

  state.SetItemsProcessed(state.iterations() * kNumTests);
}
BENCHMARK(BM_Neighbors1);

BENCHMARK_MAIN();
