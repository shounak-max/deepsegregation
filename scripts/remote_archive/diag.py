
import numpy as np, pickle, sys
from pathlib import Path
from scipy.spatial import KDTree

cache = Path('data/PSNet/cache')

def load(name):
    cf = cache / f'{name}.pkl'
    with open(cf, 'rb') as f: return pickle.load(f)

areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4']
for a in areas:
    cf = cache / f'{a}.pkl'
    if not cf.exists():
        print(f'{a}: no cache'); continue
    pts, lbl = load(a)
    class_names = ['ibeam','pipe','pump','rectangularbeam','tank']
    print(f'\n=== {a} ===')
    print(f'  Points: {len(pts):,}')
    print(f'  X: [{pts[:,0].min():.1f}, {pts[:,0].max():.1f}]  range={pts[:,0].ptp():.1f}m')
    print(f'  Y: [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]  range={pts[:,1].ptp():.1f}m')
    print(f'  Z: [{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]  range={pts[:,2].ptp():.1f}m')
    counts = np.bincount(lbl, minlength=5)
    total = counts.sum()
    for i, (cn, c) in enumerate(zip(class_names, counts)):
        print(f'  {cn:20s}: {c:8,}  ({c/total*100:.1f}%)')

# Test sphere distribution for Area_3 (val)
print('\n=== Sphere class coverage test (Area_3, in_radius=2m) ===')
cf = cache / 'Area_3.pkl'
if cf.exists():
    pts, lbl = load('Area_3')
    kd_cf = cache / 'Area_3_kd.pkl'
    if kd_cf.exists():
        with open(kd_cf,'rb') as f: tree = pickle.load(f)
        rng = np.random.default_rng(42)
        class_names = ['ibeam','pipe','pump','rectangularbeam','tank']
        for radius in [2.0, 5.0, 10.0, 20.0]:
            cls_present = 0
            for trial in range(20):
                seed = rng.integers(0, len(pts))
                idxs = tree.query_ball_point(pts[seed], r=radius)
                if len(idxs) < 10: continue
                idxs = np.array(idxs)
                sphere_lbl = lbl[idxs]
                uniq = np.unique(sphere_lbl)
                cls_present += len(uniq)
            print(f'  radius={radius:5.1f}m: avg classes per sphere = {cls_present/20:.1f}')
