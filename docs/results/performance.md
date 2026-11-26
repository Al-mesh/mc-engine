Machine: AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD, 16 logical cores, Windows 11, Python 3.15.2, numpy 2.5.5, numba 0.68.1

Heston, 100 steps, European call, N = 2^18 (best of 3):

| Model | Backend | Paths/sec | Speed-up | RNG share of time |
|---|---|---:|---:|---:|
| Heston QE | numpy | 99,952 | 1.0x | 8% |
| Heston QE | numba, 1 thread | 126,094 | 1.3x | 10% |
| Heston QE | numba, all threads | 550,524 | 5.5x | 45% |
| Heston EULER | numpy | 280,647 | 1.0x | 38% |
| Heston EULER | numba, 1 thread | 449,958 | 1.6x | 60% |
| Heston EULER | numba, all threads | 560,531 | 2.0x | 75% |

Arithmetic Asian (252 steps), GBM, N = 2^21, multiprocessing (16 chunks):

| Workers | Seconds | Paths/sec | Speed-up |
|---|---:|---:|---:|
| serial mc_price | 9.12 | 229,902 | 1.0x |
| 1 | 9.13 | 229,636 | 1.0x |
| 2 | 6.88 | 304,980 | 1.3x |
| 4 | 5.17 | 405,497 | 1.8x |
| 8 | 5.10 | 410,940 | 1.8x |
| 16 | 6.79 | 308,740 | 1.3x |
| 8 (near-empty job: pure overhead) | 2.75 | – | – |

numba first-call cost (compile or cache load): 0.11s
