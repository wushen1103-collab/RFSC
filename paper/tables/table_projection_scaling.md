# Projection scaling (CPU, five repetitions)

Memory is peak traced allocation during the call, not process RSS.

| Operator | Nodes | Directed edges | Median | p95 | Input | Peak traced allocation |
|---|---:|---:|---:|---:|---:|---:|
| Node boundary | 1,000 | 4,000 | 13.2 ms | 13.6 ms | 0.108 MiB | 0.179 MiB |
| Node boundary | 10,000 | 40,000 | 129.7 ms | 130.5 ms | 1.015 MiB | 1.666 MiB |
| Node boundary | 25,000 | 100,000 | 335.9 ms | 342.7 ms | 2.411 MiB | 4.589 MiB |
| Complete edge Hodge | 100 | 10,000 | 0.114 ms | 0.143 ms | 0.076 MiB | 0.204 MiB |
| Complete edge Hodge | 500 | 250,000 | 0.546 ms | 1.267 ms | 1.907 MiB | 2.038 MiB |
| Complete edge Hodge | 1,000 | 1,000,000 | 2.20 ms | 3.22 ms | 7.629 MiB | 7.764 MiB |
| Complete edge Hodge | 2,000 | 4,000,000 | 105.0 ms | 1,189.6 ms | 30.518 MiB | 30.660 MiB |
| Incomplete dense LS | 50 | 1,226 | 1.37 ms | 2.08 ms | 0.028 MiB | 1.028 MiB |
| Incomplete dense LS | 100 | 4,950 | 12.9 ms | 16.7 ms | 0.113 MiB | 7.731 MiB |
| Incomplete dense LS | 133 | 8,778 | 30.5 ms | 35.7 ms | 0.201 MiB | 18.087 MiB |

The incomplete implementation is adequate for the evaluated sizes but uses a
dense incidence matrix; no large-sparse-graph efficiency claim is made.

