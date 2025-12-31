window.BENCHMARK_DATA = {
  "lastUpdate": 1767145938366,
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
        "date": 1767145937353,
        "tool": "googlecpp",
        "benches": [
          {
            "name": "BM_Index1M",
            "value": 128836696.19999978,
            "unit": "ns/iter",
            "extra": "iterations: 5\ncpu: 128817124.20000008 ns\nthreads: 1"
          },
          {
            "name": "BM_Search10Percent",
            "value": 182575436.49999875,
            "unit": "ns/iter",
            "extra": "iterations: 4\ncpu: 182525298.24999985 ns\nthreads: 1"
          },
          {
            "name": "BM_Search1Percent",
            "value": 33543297.333333537,
            "unit": "ns/iter",
            "extra": "iterations: 21\ncpu: 33540308.095238134 ns\nthreads: 1"
          },
          {
            "name": "BM_Search001Percent",
            "value": 3141682.303964749,
            "unit": "ns/iter",
            "extra": "iterations: 227\ncpu: 3141270.3348017535 ns\nthreads: 1"
          },
          {
            "name": "BM_Neighbors100",
            "value": 18854832.405405413,
            "unit": "ns/iter",
            "extra": "iterations: 37\ncpu: 18853739.999999978 ns\nthreads: 1"
          },
          {
            "name": "BM_NeighborsAll",
            "value": 113941605.16666573,
            "unit": "ns/iter",
            "extra": "iterations: 6\ncpu: 113934580.3333338 ns\nthreads: 1"
          },
          {
            "name": "BM_Neighbors1",
            "value": 561949709.0000038,
            "unit": "ns/iter",
            "extra": "iterations: 1\ncpu: 561931339.9999975 ns\nthreads: 1"
          }
        ]
      }
    ]
  }
}