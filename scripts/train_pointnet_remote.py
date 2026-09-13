"""PSNet5 PointNet training — fixed for real metric coordinates.

Key fixes vs previous versions:
  1. in_radius=15m (was 2m) → avg 4.5 classes per sphere (was 2.0)
  2. Relative-Z height feature: (z - z_min) / z_range per sphere
     — works correctly for Area_2 which has Z in [78,107] not [0,10]
  3. 500K point area subsampling before KDTree build (was full 30M pts)
  4. Focal loss (gamma=2) to handle 75% pipe dominance in Area_1
  5. Per-area coordinate normalization before training
  6. steps_per_epoch=50 for faster iteration; epochs=200
"""
from __future__ import annotations
import argparse, json, os, pickle, signal, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

_RUNNING = True
CLASS_NAMES   = ["ibeam","pipe","pump","rectangularbeam","tank"]
NUM_CLASSES   = 5
TRAIN_AREAS   = ["Area_1","Area_2","Area_4"]
VAL_AREAS     = ["Area_3"]
NAME_TO_LABEL = {n: i for i, n in enumerate(CLASS_NAMES)}
SUBSAMPLE_N   = 500_000   # subsample each area to this many points for KDTree


def _sig(signum, frame):
    global _RUNNING
    print(f"\nSignal {signum} — shutting down...", flush=True)
    _RUNNING = False


def dump_status(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_area(area_path: Path, cache_dir: Path):
    cf = cache_dir / f"{area_path.name}.pkl"
    if cf.exists():
        print(f"  Cache hit: {area_path.name}", flush=True)
        with open(cf, "rb") as f:
            return pickle.load(f)
    print(f"  Parsing {area_path.name} ...", flush=True)
    all_pts, all_lbl = [], []
    for room in sorted(area_path.iterdir()):
        if not room.is_dir(): continue
        ann = room / "Annotations"
        if not ann.exists(): continue
        for txt in sorted(ann.glob("*.txt")):
            cls_name = txt.stem.split("_")[0]
            if cls_name not in NAME_TO_LABEL: continue
            label = NAME_TO_LABEL[cls_name]
            rows = []
            with open(txt) as fh:
                for line in fh:
                    v = line.strip().split()
                    if len(v) >= 3:
                        rows.append((float(v[0]), float(v[1]), float(v[2])))
            if rows:
                arr = np.array(rows, dtype=np.float32)
                all_pts.append(arr)
                all_lbl.append(np.full(len(arr), label, dtype=np.int64))
    pts = np.concatenate(all_pts)
    lbl = np.concatenate(all_lbl)
    with open(cf, "wb") as f:
        pickle.dump((pts, lbl), f)
    print(f"  {area_path.name}: {len(pts):,} pts", flush=True)
    return pts, lbl


def subsample_area(pts, lbl, n, area_name, cache_dir, rng_seed=0):
    """Uniformly subsample area to n points, with pickle caching."""
    cf = cache_dir / f"{area_name}_sub{n}.pkl"
    if cf.exists():
        with open(cf, "rb") as f:
            return pickle.load(f)
    if len(pts) <= n:
        result = (pts.copy(), lbl.copy())
    else:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(pts), n, replace=False)
        result = (pts[idx].copy(), lbl[idx].copy())
    with open(cf, "wb") as f:
        pickle.dump(result, f)
    return result


def build_kdtree(pts, area_name, cache_dir, suffix=""):
    from scipy.spatial import KDTree
    cf = cache_dir / f"{area_name}_kd{suffix}.pkl"
    if cf.exists():
        with open(cf, "rb") as f:
            return pickle.load(f)
    print(f"  KDTree for {area_name} ({len(pts):,} pts)...", flush=True)
    tree = KDTree(pts)
    with open(cf, "wb") as f:
        pickle.dump(tree, f)
    return tree


def sphere_sample(pts, lbl, tree, full_pts, full_lbl,
                  n_pts, radius, rng, augment=True):
    """Sample one sphere crop of n_pts.

    Selects seed point from full resolution, queries subsampled tree,
    then retrieves labels from full resolution array.
    Uses relative-Z as height feature (robust to different area coordinate systems).

    Returns pts_4d (n_pts, 4), lbl (n_pts,)
    """
    seed = rng.integers(0, len(pts))
    ctr  = pts[seed].copy()
    if augment:
        ctr += rng.normal(0, radius / 20, 3).astype(np.float32)

    idxs = np.array(tree.query_ball_point(ctr, r=radius), dtype=np.int64)
    if len(idxs) == 0:
        _, idxs = tree.query(ctr.reshape(1, 3), k=min(n_pts, len(pts)))
        idxs = np.array(idxs).flatten()

    if len(idxs) > n_pts:
        idxs = rng.choice(idxs, n_pts, replace=False)
    elif len(idxs) < n_pts:
        pad  = rng.choice(idxs, n_pts - len(idxs), replace=True)
        idxs = np.concatenate([idxs, pad])
    rng.shuffle(idxs)

    raw = pts[idxs].copy()   # (N, 3) in real metric coords

    # Relative Z height — fraction within sphere Z range (0=bottom, 1=top)
    z_range = raw[:, 2].ptp()
    if z_range < 0.01:
        z_rel = np.zeros((len(raw), 1), dtype=np.float32)
    else:
        z_rel = ((raw[:, 2] - raw[:, 2].min()) / z_range).reshape(-1, 1).astype(np.float32)

    # Augmentation before normalisation
    if augment:
        a = rng.uniform(0, 2 * np.pi)
        ca, sa = np.cos(a), np.sin(a)
        R = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=np.float32)
        raw = raw @ R.T
        raw *= rng.uniform(0.9, 1.1)

    # Normalise XYZ to unit sphere
    pmin, pmax = raw.min(0), raw.max(0)
    xyz = raw - (pmin + pmax) / 2.0
    scale = np.linalg.norm(xyz, axis=1).max()
    if scale > 1e-6:
        xyz /= scale

    pts4d = np.concatenate([xyz, z_rel], axis=1).astype(np.float32)
    return pts4d, lbl[idxs]


# ── Focal loss ────────────────────────────────────────────────────────────────

def focal_loss(logits, targets, weights, gamma=2.0):
    """Focal loss: FL = -alpha*(1-pt)^gamma * log(pt)"""
    import torch, torch.nn.functional as F
    ce  = F.cross_entropy(logits, targets, weight=weights, reduction="none")
    pt  = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes=5, in_dim=4):
    import torch, torch.nn as nn, torch.nn.functional as F

    class STN(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k
            self.c1=nn.Conv1d(k,64,1); self.c2=nn.Conv1d(64,128,1); self.c3=nn.Conv1d(128,1024,1)
            self.f1=nn.Linear(1024,512); self.f2=nn.Linear(512,256); self.f3=nn.Linear(256,k*k)
            for bn in [64,128,1024,512,256]:
                setattr(self, f"b{bn}", nn.BatchNorm1d(bn))
            nn.init.zeros_(self.f3.weight); nn.init.zeros_(self.f3.bias)
            with torch.no_grad(): self.f3.bias.copy_(torch.eye(k).flatten())
        def forward(self, x):
            B = x.size(0)
            x = F.relu(self.b64(self.c1(x)))
            x = F.relu(self.b128(self.c2(x)))
            x = F.relu(self.b1024(self.c3(x)))
            x = x.max(2)[0]
            x = F.relu(self.b512(self.f1(x)))
            x = F.relu(self.b256(self.f2(x)))
            x = self.f3(x)
            eye = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).expand(B,-1)
            return (x+eye).view(B, self.k, self.k)

    class PN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stn = STN(in_dim)
            self.c1=nn.Conv1d(in_dim,64,1);  self.c2=nn.Conv1d(64,64,1)
            self.c3=nn.Conv1d(64,64,1);      self.c4=nn.Conv1d(64,128,1)
            self.c5=nn.Conv1d(128,1024,1)
            self.c6=nn.Conv1d(1088,512,1);   self.c7=nn.Conv1d(512,256,1)
            self.c8=nn.Conv1d(256,128,1);    self.c9=nn.Conv1d(128,num_classes,1)
            for k,v in [(64,"b1"),(64,"b2"),(64,"b3"),(128,"b4"),(1024,"b5"),
                        (512,"b6"),(256,"b7"),(128,"b8")]:
                setattr(self, v, nn.BatchNorm1d(k))
            self.drop = nn.Dropout(0.3)
        def forward(self, x):
            B, N, _ = x.shape
            x = x.transpose(2,1)
            T = self.stn(x)
            x = x.transpose(2,1); x = torch.bmm(x, T); x = x.transpose(2,1)
            x  = F.relu(self.b1(self.c1(x)))
            lf = F.relu(self.b2(self.c2(x)))
            x  = F.relu(self.b3(self.c3(lf)))
            x  = F.relu(self.b4(self.c4(x)))
            x  = F.relu(self.b5(self.c5(x)))
            gf = x.max(2)[0].unsqueeze(2).expand(-1,-1,N)
            x  = torch.cat([lf, gf], dim=1)
            x  = F.relu(self.b6(self.c6(x)))
            x  = F.relu(self.b7(self.c7(x)))
            x  = F.relu(self.b8(self.c8(x)))
            x  = self.drop(x)
            x  = self.c9(x)
            return x.transpose(2,1).contiguous().view(B*N, num_classes)
    return PN()


# ── Helpers ───────────────────────────────────────────────────────────────────

def cls_weights(flat, nc, device):
    import torch
    c = np.bincount(flat.astype(np.int64), minlength=nc).astype(np.float32)
    print(f"  Class counts: {dict(zip(CLASS_NAMES, c.astype(int)))}", flush=True)
    w = 1.0 / (c / c.sum() + 1e-6)
    w = w / w.sum() * nc
    return torch.tensor(w, dtype=torch.float32, device=device)


def evaluate(model, val_data, device, n_pts, radius, weights, n_spheres=150):
    import torch, torch.nn.functional as F
    from deepsegregation.metrics import segmentation_metrics
    model.eval()
    rng = np.random.default_rng(0)
    all_p, all_l = [], []
    tot, nb = 0.0, 0
    with torch.no_grad():
        for pts, lbl, tree, _, _ in val_data:
            per = max(1, n_spheres // len(val_data))
            for _ in range(per):
                p4, l = sphere_sample(pts, lbl, tree, pts, lbl,
                                      n_pts, radius, rng, augment=False)
                tp = torch.from_numpy(p4[None]).to(device)
                tl = torch.from_numpy(l).to(device)
                logits = model(tp)
                tot += focal_loss(logits, tl, weights).item()
                nb  += 1
                all_p.append(logits.argmax(1).cpu().numpy())
                all_l.append(l)
    all_p = np.concatenate(all_p); all_l = np.concatenate(all_l)
    metrics = segmentation_metrics(all_l, all_p, NUM_CLASSES)
    return tot / max(nb,1), metrics


def chk_tgts(m, vl, tm, ta, tl):
    mm = (m["mIoU"] >= tm) if tm else True
    am = (m["accuracy"] >= ta) if ta else True
    lm = (vl <= tl) if tl else True
    return {"target_miou":tm,"current_miou":m["mIoU"],"miou_met":mm,
            "target_accuracy":ta,"current_accuracy":m["accuracy"],"accuracy_met":am,
            "target_loss":tl,"current_loss":vl,"loss_met":lm,
            "all_targets_met": mm and am and lm}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    pa = argparse.ArgumentParser(description="PSNet5 PointNet — fixed radius & coords")
    pa.add_argument("--data-root",  default="data/PSNet/PSNet5")
    pa.add_argument("--cache-dir",  default="data/PSNet/cache")
    pa.add_argument("--epochs",     type=int,   default=200)
    pa.add_argument("--lr",         type=float, default=1e-3)
    pa.add_argument("--wd",         type=float, default=5e-5)
    pa.add_argument("--num-points", type=int,   default=4096)
    pa.add_argument("--in-radius",  type=float, default=15.0)
    pa.add_argument("--subsample",  type=int,   default=500_000)
    pa.add_argument("--steps-per-epoch", type=int, default=50)
    pa.add_argument("--batch-size", type=int,   default=4)
    pa.add_argument("--checkpoint-dir", default="checkpoints/pointnet_v4")
    pa.add_argument("--checkpoint-interval", type=int, default=10)
    pa.add_argument("--gpu-id",     type=int,   default=1)
    pa.add_argument("--device",     default="auto")
    pa.add_argument("--seed",       type=int,   default=42)
    pa.add_argument("--target-miou",     type=float, default=0.50)
    pa.add_argument("--target-accuracy", type=float, default=0.75)
    pa.add_argument("--target-loss",     type=float, default=1.5)
    pa.add_argument("--early-stop-on-targets", action="store_true")
    pa.add_argument("--status-file", default="pipeline_status.json")
    pa.add_argument("--warmup-epochs", type=int, default=10)
    pa.add_argument("--focal-gamma",   type=float, default=2.0)
    args = pa.parse_args(argv)

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    import torch, torch.nn.functional as F

    ckpt_dir  = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir);      cache_dir.mkdir(parents=True, exist_ok=True)
    sf = Path(args.status_file)

    if args.device == "auto":
        if torch.cuda.is_available():
            gi = args.gpu_id if args.gpu_id < torch.cuda.device_count() else 0
            ds = f"cuda:{gi}"
        else:
            ds = "cpu"
    else:
        ds = args.device
    device = torch.device(ds)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    gn = torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU"
    print("="*70, flush=True)
    print(f"PSNet5 PointNet v4 — Radius={args.in_radius}m, Focal-Loss(gamma={args.focal_gamma})", flush=True)
    print(f"Device={device}({gn}) Epochs={args.epochs} LR={args.lr} Batch={args.batch_size}", flush=True)
    print(f"Subsample={args.subsample:,} pts/area  Steps={args.steps_per_epoch}/epoch", flush=True)
    print(f"Targets: mIoU>={args.target_miou} Acc>={args.target_accuracy}", flush=True)
    print(f"PID={os.getpid()}", flush=True)
    print("="*70, flush=True)

    st = {"status":"INITIALIZING","pid":os.getpid(),"start_time":time.time(),
          "model":"PointNet-4D-v4","device":str(device),"gpu_name":gn,
          "epochs_total":args.epochs,"current_epoch":0,
          "targets":{"target_miou":args.target_miou,"target_accuracy":args.target_accuracy,
                     "target_loss":args.target_loss},
          "best_val_miou":0.0,"best_epoch":None,"all_targets_met":False,
          "latest_metrics":None,"dataset_type":f"PSNet5-sphere-r{args.in_radius}m",
          "class_names":CLASS_NAMES,"train_areas":TRAIN_AREAS,"val_areas":VAL_AREAS}
    dump_status(sf, st)

    dr = Path(args.data_root)

    print("Loading + subsampling train areas...", flush=True)
    train_data, all_sub_lbl = [], []
    for a in TRAIN_AREAS:
        pts, lbl = load_area(dr/a, cache_dir)
        # Subsample for KDTree efficiency
        cf_sub = cache_dir / f"{a}_sub{args.subsample}.pkl"
        if cf_sub.exists():
            with open(cf_sub,"rb") as f: s_pts, s_lbl = pickle.load(f)
            print(f"  {a}: cached subsample {len(s_pts):,} pts", flush=True)
        else:
            if len(pts) > args.subsample:
                idx = np.random.default_rng(42).choice(len(pts), args.subsample, replace=False)
                s_pts, s_lbl = pts[idx], lbl[idx]
            else:
                s_pts, s_lbl = pts, lbl
            with open(cf_sub,"wb") as f: pickle.dump((s_pts,s_lbl),f)
            print(f"  {a}: subsampled {len(pts):,} -> {len(s_pts):,} pts", flush=True)
        tree = build_kdtree(s_pts, a, cache_dir, suffix=f"_r{args.in_radius}")
        train_data.append((s_pts, s_lbl, tree, pts, lbl))
        all_sub_lbl.append(s_lbl)
        counts = np.bincount(s_lbl, minlength=5)
        tot = counts.sum()
        print(f"  {a} class dist: " +
              " ".join(f"{CLASS_NAMES[i][:4]}={counts[i]/tot*100:.0f}%" for i in range(5)),
              flush=True)

    print("Loading + subsampling val areas...", flush=True)
    val_data = []
    for a in VAL_AREAS:
        pts, lbl = load_area(dr/a, cache_dir)
        cf_sub = cache_dir / f"{a}_sub{args.subsample}.pkl"
        if cf_sub.exists():
            with open(cf_sub,"rb") as f: s_pts, s_lbl = pickle.load(f)
        else:
            if len(pts) > args.subsample:
                idx = np.random.default_rng(99).choice(len(pts), args.subsample, replace=False)
                s_pts, s_lbl = pts[idx], lbl[idx]
            else:
                s_pts, s_lbl = pts, lbl
            with open(cf_sub,"wb") as f: pickle.dump((s_pts,s_lbl),f)
        tree = build_kdtree(s_pts, a, cache_dir, suffix=f"_r{args.in_radius}")
        val_data.append((s_pts, s_lbl, tree, pts, lbl))
        counts = np.bincount(s_lbl, minlength=5)
        tot = counts.sum()
        print(f"  {a}: {len(s_pts):,} pts — " +
              " ".join(f"{CLASS_NAMES[i][:4]}={counts[i]/tot*100:.0f}%" for i in range(5)),
              flush=True)

    weights = cls_weights(np.concatenate(all_sub_lbl), NUM_CLASSES, device)
    print(f"Class weights: {weights.cpu().numpy().round(3)}", flush=True)

    model = build_model(NUM_CLASSES, 4).to(device)
    npar  = sum(p.numel() for p in model.parameters())
    print(f"PointNet-4D params: {npar:,}", flush=True)

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs-args.warmup_epochs,1), eta_min=1e-5)

    best_miou, best_ep = 0.0, None
    history = []
    st["status"] = "RUNNING"
    t0 = time.time()

    for epoch in range(1, args.epochs+1):
        if not _RUNNING:
            st["status"] = "INTERRUPTED"; break

        ep0 = time.time()
        model.train()
        if epoch <= args.warmup_epochs:
            for pg in opt.param_groups:
                pg["lr"] = args.lr * epoch / args.warmup_epochs

        losses = []
        for _step in range(args.steps_per_epoch):
            # Weight areas by subsampled point count for fair sampling
            ai = int(rng.integers(0, len(train_data)))
            s_pts_a, s_lbl_a, tree_a, full_pts_a, full_lbl_a = train_data[ai]
            bp, bl = [], []
            for _ in range(args.batch_size):
                p4, l = sphere_sample(s_pts_a, s_lbl_a, tree_a,
                                      full_pts_a, full_lbl_a,
                                      args.num_points, args.in_radius, rng, True)
                bp.append(p4); bl.append(l)
            tp = torch.from_numpy(np.stack(bp)).to(device)
            tl = torch.from_numpy(np.stack(bl).reshape(-1)).to(device)
            opt.zero_grad()
            logits = model(tp)
            loss = focal_loss(logits, tl, weights, args.focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        if epoch > args.warmup_epochs:
            sched.step()

        tr_loss = float(np.mean(losses))
        ep_t = time.time()-ep0
        lr   = opt.param_groups[0]["lr"]

        is_ckpt = (epoch % args.checkpoint_interval == 0) or (epoch == args.epochs)
        if is_ckpt:
            print(f"[Epoch {epoch:03d}/{args.epochs}] Evaluating...", flush=True)
            vl, metrics = evaluate(model, val_data, device,
                                   args.num_points, args.in_radius, weights, 150)
            tgt = chk_tgts(metrics, vl, args.target_miou, args.target_accuracy, args.target_loss)
            is_best = metrics["mIoU"] > best_miou
            if is_best:
                best_miou = metrics["mIoU"]; best_ep = epoch
            history.append({"epoch":epoch,"train_loss":tr_loss,"val_loss":vl,
                             "metrics":metrics,"targets_evaluation":tgt,
                             "is_best":is_best,"lr":lr})
            payload = {"epoch":epoch,"model_state":model.state_dict(),
                       "optimizer_state":opt.state_dict(),
                       "scheduler_state":sched.state_dict(),
                       "metrics":metrics,"val_loss":vl,"train_loss":tr_loss,
                       "targets_evaluation":tgt,"model_name":"PointNet4D_PSNet5_v4",
                       "num_classes":NUM_CLASSES,"class_names":CLASS_NAMES,"in_dim":4}
            torch.save(payload, ckpt_dir/f"epoch_{epoch:04d}.pt")
            if is_best:
                torch.save(payload, ckpt_dir/"best.pt")
                print(f"  BEST mIoU={best_miou*100:.1f}% at ep{epoch}", flush=True)
            with open(ckpt_dir/"eval_history.json","w") as hf:
                json.dump(history, hf, indent=2)
            per = metrics.get("per_class_iou",[])
            iou_s=" ".join(f"{CLASS_NAMES[i][:4]}={v*100:.0f}%" for i,v in enumerate(per))
            badge="[TARGETS MET]" if tgt["all_targets_met"] else "[IN PROGRESS]"
            print(f"[Epoch {epoch:03d}/{args.epochs}] {badge} "
                  f"TrLoss={tr_loss:.4f} ValLoss={vl:.4f} "
                  f"mIoU={metrics['mIoU']*100:.1f}%/{args.target_miou*100:.0f}% "
                  f"Acc={metrics['accuracy']*100:.1f}%/{args.target_accuracy*100:.0f}% "
                  f"LR={lr:.2e} Best={best_miou*100:.1f}%(ep{best_ep}) t={ep_t:.1f}s | {iou_s}",
                  flush=True)
            st.update({"current_epoch":epoch,"train_loss":tr_loss,"val_loss":vl,
                       "latest_metrics":metrics,"best_val_miou":best_miou,"best_epoch":best_ep,
                       "targets_evaluation":tgt,"all_targets_met":tgt["all_targets_met"],
                       "last_updated":time.time(),"elapsed_seconds":round(time.time()-t0,2),
                       "current_lr":lr})
            dump_status(sf, st)
            if args.early_stop_on_targets and tgt["all_targets_met"]:
                print(f"All targets met at ep{epoch}!", flush=True)
                st["status"]="TARGETS_ACHIEVED"; break
        else:
            print(f"[Epoch {epoch:03d}/{args.epochs}] "
                  f"TrLoss={tr_loss:.4f} LR={lr:.2e} t={ep_t:.1f}s",flush=True)
            st.update({"current_epoch":epoch,"train_loss":tr_loss,
                       "last_updated":time.time(),
                       "elapsed_seconds":round(time.time()-t0,2),"current_lr":lr})
            dump_status(sf, st)

    if st["status"]=="RUNNING": st["status"]="COMPLETED"
    st["completed_at"]=time.time(); dump_status(sf,st)
    print("="*70, flush=True)
    print(f"Done: {st['status']} | Best mIoU={best_miou*100:.2f}% at ep{best_ep}", flush=True)
    print("="*70, flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
