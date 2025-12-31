window.BENCHMARK_DATA = {
  "lastUpdate": 1767145945848,
  "repoUrl": "https://github.com/chusitoo/flatbush",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "chusitoo",
            "username": "chusitoo"
          },
          "committer": {
            "name": "chusitoo",
            "username": "chusitoo"
          },
          "id": "bc865358cfcda90be4c877908618eadf9c709d3c",
          "message": "Segregate compiler benchmarks into subfolders",
          "timestamp": "2025-12-31T00:41:49Z",
          "url": "https://github.com/chusitoo/flatbush/pull/7/commits/bc865358cfcda90be4c877908618eadf9c709d3c"
        },
        "date": 1767145944969,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_Index1M",
            "value": 131322363.39999965,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 131306559.60000011 ns\nthreads: 1"
          },
          {
            "name": "BM_Search10Percent",
            "value": 173037875.5000004,
            "unit": "ns/iter",
            "extra": "iterations: 4\ncpu: 173032163.24999982 ns\nthreads: 1"
          },
          {
            "name": "BM_Search1Percent",
            "value": 29955941.72727258,
            "unit": "ns/iter",
            "extra": "iterations: 22\ncpu: 29951497.954545498 ns\nthreads: 1"
          },
          {
            "name": "BM_Search001Percent",
            "value": 3051587.4008810166,
            "unit": "ns/iter",
            "extra": "iterations: 227\ncpu: 3051456.1718061618 ns\nthreads: 1"
          },
          {
            "name": "BM_Neighbors100",
            "value": 16749213.642856916,
            "unit": "ns/iter",
            "extra": "iterations: 42\ncpu: 16747471.04761903 ns\nthreads: 1"
          },
          {
            "name": "BM_NeighborsAll",
            "value": 108772158.99999726,
            "unit": "ns/iter",
            "extra": "iterations: 6\ncpu: 108763366.49999984 ns\nthreads: 1"
          },
          {
            "name": "BM_Neighbors1",
            "value": 470592468.0000067,
            "unit": "ns/iter",
            "extra": "iterations: 2\ncpu: 470531825.5000002 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}