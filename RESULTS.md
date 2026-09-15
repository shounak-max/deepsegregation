# Verified Progress Results

These results were produced on the remote Tesla K80 using the saved pure
PyTorch PointNet++ SSG binary checkpoint:

`checkpoints/step4_cellparity_diag/best.pt`

## Semantic segmentation

| Scene | Accuracy | mIoU | Pipe IoU | Pipe precision | Pipe recall | Pipe F1 | Points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Area 4 validation | 79.56% | 61.23% | 47.55% | 59.47% | 70.34% | 64.45% | 11,585,960 |
| Area 3 held-out test | 68.26% | 47.62% | 32.79% | 54.65% | 45.05% | 49.39% | 24,466,092 |

The Area 3 scene was not used for training, checkpoint selection, or
threshold tuning. Both evaluations used 0.5 m spatial cells, complete-cell
normalization, and 2048-point inference chunks.

## Preliminary geometry extraction

The Area 3 checkpoint predicted 6,930,685 pipe points. A diagnostic DBSCAN
run on a random 300,000-point sample produced 2,544 candidate clusters and
319 accepted cylinder fits after minimum-size and RANSAC consensus checks.

The reversible-coordinate audit recorded:

- scene scale: `31.6903`;
- median fitted radius: `0.1227` original coordinate units;
- 10th-90th percentile radius: `0.0350-0.3610`;
- median inlier ratio: `86.2%`;
- median cylinder RMSE: `0.00484` original coordinate units.

These are preliminary geometric candidates, not a complete pipe inventory or
ground-truth diameter measurements. Area 4 geometry calibration and full
spatial clustering remain necessary before interpreting them as physical pipe
objects.

## Limitations

- The model is a pure PyTorch PointNet++ SSG baseline, not ResPointNet++.
- PSNet5 semantic labels do not provide pipe instance IDs or diameters.
- The upstream custom CUDA implementation remains postponed.
- Radius estimates require confidence gates and reversible coordinate metadata.
