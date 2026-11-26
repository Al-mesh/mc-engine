| Problem | Technique | Price | SE | VR ratio | Efficiency gain |
|---|---|---:|---:|---:|---:|
| European call, GBM (1 step) | plain | 10.4254 | 1.44e-02 | 1.0 | 1.0 |
| European call, GBM (1 step) | antithetic | 10.4314 | 1.02e-02 | 2.0 | 2.3 |
| European call, GBM (1 step) | control (S_T) | 10.4342 | 5.48e-03 | 6.9 | 6.5 |
| European call, GBM (1 step) | RQMC | 10.4506 | 3.80e-05 | 142,378.0 | 96,116.1 |
| Arithmetic Asian, GBM (12 fixings) | plain | 6.1583 | 8.33e-03 | 1.0 | 1.0 |
| Arithmetic Asian, GBM (12 fixings) | antithetic | 6.1598 | 5.77e-03 | 2.1 | 2.4 |
| Arithmetic Asian, GBM (12 fixings) | control (geometric Asian) | 6.1564 | 2.32e-04 | 1,283.4 | 1,003.7 |
| Arithmetic Asian, GBM (12 fixings) | RQMC (forward) | 6.1556 | 6.01e-04 | 191.7 | 134.4 |
| Arithmetic Asian, GBM (12 fixings) | RQMC + Brownian bridge | 6.1562 | 1.52e-04 | 3,012.5 | 1,423.5 |
| Up-and-out call B=130, GBM (50 dates) | plain | 3.8180 | 6.50e-03 | 1.0 | 1.0 |
| Up-and-out call B=130, GBM (50 dates) | antithetic | 3.8235 | 5.38e-03 | 1.5 | 1.6 |
| Up-and-out call B=130, GBM (50 dates) | RQMC + Brownian bridge | 3.8223 | 1.77e-03 | 13.4 | 6.3 |
| European call, Heston QE (50 steps) | plain | 8.6022 | 1.86e-02 | 1.0 | 1.0 |
| European call, Heston QE (50 steps) | antithetic | 8.5699 | 1.39e-02 | 1.8 | 1.9 |
| European call, Heston QE (50 steps) | control (S_T) | 8.5885 | 1.08e-02 | 3.0 | 2.8 |
| European call, Heston QE (50 steps) | RQMC | 8.5881 | 8.91e-03 | 4.4 | 5.8 |
