# Empirical Baseline & Ablation Study Results

Three-way information-equal ablation: Unconstrained vs. Prior-Initialized (same axis info, no hard constraint) vs. Topological (same axis info + C1 constraint).

| Noise (mm) | Occlusion | Unconstrained Bend Err | Prior-Init Bend Err | Topological Bend Err | Unconstrained RMSE | Prior-Init RMSE | Topological RMSE | Speedup (vs Unconstrained) |
|---|---|---|---|---|---|---|---|---|
| 0.0 mm | 0% | 0.00% | 0.00% | **0.00%** | 0.000 mm | 0.000 mm | **0.000 mm** | **143.3x** |
| 0.0 mm | 30% | 0.00% | 0.00% | **0.00%** | 0.000 mm | 0.000 mm | **0.000 mm** | **220.0x** |
| 2.0 mm | 0% | 0.04% | 0.04% | **0.03%** | 1.904 mm | 1.904 mm | **1.915 mm** | **119.7x** |
| 2.0 mm | 30% | 0.40% | 0.40% | **0.01%** | 1.867 mm | 1.867 mm | **1.873 mm** | **190.7x** |
| 5.0 mm | 0% | 0.06% | 0.06% | **0.08%** | 4.628 mm | 4.628 mm | **4.662 mm** | **123.7x** |
| 5.0 mm | 30% | 1.55% | 1.55% | **0.03%** | 4.474 mm | 4.474 mm | **4.486 mm** | **184.6x** |
