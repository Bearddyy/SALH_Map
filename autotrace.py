#!/usr/bin/env python3
"""
autotrace.py - automatically cut a tithe map into candidate field polygons.

    python autotrace.py MAP.jpg                # -> auto_cells.json + auto_preview.jpg
    python autotrace.py MAP.jpg --scale 0.5    # working resolution (0.5 = same as tracer_app.py)

How it works (all classical image processing, no training data needed):
  1. Ink  - a ridge filter (Sato) picks out thin dark lines whatever the paper tone,
            hysteresis keeps faint line sections that join up with strong ones.
  2. Lines- only long connected ink is treated as boundary; small blobs (the plot
            numbers, lettering) are set aside and remembered as 'seeds'.
  3. Cells- every pixel is assigned to a cell with a watershed on the distance-to-ink
            map.  The cell edge therefore lands ON the drawn line (no inset), and a
            small gap in a line cannot leak because neighbouring cells compete.
  4. Merge- watershed over-splits big fields; two neighbours are merged again when the
            cut between them runs through blank paper instead of along ink.
"""
import argparse, json, os, sys, time
import numpy as np, cv2
from scipy import ndimage as ndi
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

P = dict(
    sigmas=(0.8, 1.2, 1.8),   # line widths to look for, in working-res px
    lo=0.025, hi=0.08,        # hysteresis thresholds on ridge strength (lower = finds fainter lines, more noise)
    line_diag=30, line_area=250,   # ink components bigger than this are 'line network'
    h=1.5,                    # peak dynamics for watershed seeds (px of distance)
    merge_paper=0.50,         # merge if >50% of the cut runs through blank paper
    min_area=60,              # ignore cells smaller than this (px at working res)
    frame_margin=40,          # cells touching this outer band are the page margin
    eps=1.2,                  # polygon simplification (px)
    max_frac=0.03,            # cells bigger than this fraction of the page are the blank margin, not a field
)

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ------------------------------------------------------------------ 1. ink
def ink_mask(gray):
    g = gray.astype(np.float32)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    bg = cv2.GaussianBlur(bg, (0, 0), 8)
    dark = 1 - np.clip(g / np.maximum(bg, 1), 0, 1.2)
    r = sato(dark, sigmas=P["sigmas"], black_ridges=False).astype(np.float32)
    return r

def split_lines_glyphs(r):
    ink = apply_hysteresis_threshold(r, P["lo"], P["hi"]).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    diag = np.hypot(st[:, 2], st[:, 3])
    isline = np.zeros(n, bool)
    isline[1:] = (diag[1:] >= P["line_diag"]) | (st[1:, 4] >= P["line_area"])
    lines = isline[lab]
    return lines, ink

def number_seeds(r, lines):
    """Small, faint blobs that are not part of the line network = digits & text."""
    ink_s = apply_hysteresis_threshold(r, 0.03, 0.07).astype(np.uint8)
    ink_s[cv2.dilate(lines.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink_s, connectivity=8)
    ok = np.zeros(n, bool)
    ok[1:] = (st[1:, 4] >= 6) & (st[1:, 4] <= 400) & (st[1:, 3] <= 22) & (st[1:, 2] <= 26)
    g = ok[lab].astype(np.uint8)
    cl = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7)))
    n2, l2, st2, _ = cv2.connectedComponentsWithStats(cl, connectivity=8)
    seeds = []
    for i in range(1, n2):
        if st2[i, 4] > 4000: continue          # a giant cluster = lettering / texture, not a number
        x, y, w, h = st2[i, :4]
        sub = (l2[y:y+h, x:x+w] == i) & (g[y:y+h, x:x+w] > 0)
        ys, xs = np.nonzero(sub)
        if len(xs) == 0: continue
        xs = xs + x; ys = ys + y
        seeds.append(dict(x=float(xs.mean()), y=float(ys.mean()),
                          bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], n=int(len(xs))))
    return seeds

# ------------------------------------------------------------------ 3/4. cells
def cut_stats(lab, D):
    """for every pair of touching labels: (#cut pixel pairs, #of them lying in blank paper)"""
    N = int(lab.max()) + 1
    tot, pap = {}, {}
    keys_all, paper_all = [], []
    for a, b, da, db in ((lab[:, :-1], lab[:, 1:], D[:, :-1], D[:, 1:]),
                         (lab[:-1, :], lab[1:, :], D[:-1, :], D[1:, :])):
        m = a != b
        x, y = a[m].astype(np.int64), b[m].astype(np.int64)
        lo, hi = np.minimum(x, y), np.maximum(x, y)
        keys_all.append(lo * N + hi)
        paper_all.append(np.minimum(da[m], db[m]) >= 1.0)
    keys = np.concatenate(keys_all); paper = np.concatenate(paper_all)
    uk, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv); pp = np.bincount(inv, weights=paper)
    return uk // N, uk % N, cnt, pp

def merge_regions(ws, D):
    parent = np.arange(int(ws.max()) + 1)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    lab = ws
    for it in range(8):
        a, b, cnt, pp = cut_stats(lab, D)
        frac = pp / np.maximum(cnt, 1)
        sel = np.nonzero(frac > P["merge_paper"])[0]
        if len(sel) == 0: break
        for i in sel[np.argsort(-frac[sel])]:
            ra, rb = find(a[i]), find(b[i])
            if ra != rb: parent[rb] = ra
        roots = np.array([find(i) for i in range(len(parent))])
        lab = roots[ws]
        log(f"  merge pass {it+1}: merged {len(sel)} cuts -> {len(np.unique(lab))} regions")
    return lab

def build(img_bgr, scale):
    if scale != 1.0:
        img = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img = img_bgr
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    log(f"working image {W}x{H}")
    r = ink_mask(gray); log("ridge filter done")
    lines, _ = split_lines_glyphs(r)
    seeds = number_seeds(r, lines); log(f"ink lines found; {len(seeds)} number-like blobs")
    free = (~lines).astype(np.uint8)
    D = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    Ds = cv2.GaussianBlur(D, (0, 0), 1.0)
    pk = h_maxima(Ds, P["h"])
    markers, nm = ndi.label(pk)
    log(f"{nm} watershed seeds")
    ws = watershed(-Ds, markers)
    log("watershed done")
    reg = merge_regions(ws, D)
    return img, reg, seeds, Ds

def polygons(reg, seeds, Ds, W, H):
    ids, counts = np.unique(reg, return_counts=True)
    sx = np.clip(np.array([int(round(s["x"])) for s in seeds]), 0, W - 1)
    sy = np.clip(np.array([int(round(s["y"])) for s in seeds]), 0, H - 1)
    seed_reg = reg[sy, sx] if len(seeds) else np.array([], int)
    objs = ndi.find_objects(reg)  # index = label-1
    m = P["frame_margin"]
    border = set(np.unique(np.concatenate([reg[:m].ravel(), reg[-m:].ravel(), reg[:, :m].ravel(), reg[:, -m:].ravel()])).tolist())
    keep = [int(r) for r, a in zip(ids, counts) if r != 0 and r not in border and P["min_area"] <= a <= P["max_frac"] * W * H]
    poles = ndi.maximum_position(Ds, labels=reg, index=keep)
    area_of = dict(zip(ids.tolist(), counts.tolist()))
    cells = []
    for rid, pole in zip(keep, poles):
        sl = objs[rid - 1]
        if sl is None: continue
        y0, x0 = sl[0].start, sl[1].start
        mk = (reg[sl] == rid).astype(np.uint8)
        mk = cv2.copyMakeBorder(mk, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs: continue
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, P["eps"], True).reshape(-1, 2) - 1
        if len(c) < 3: continue
        mine = np.nonzero(seed_reg == rid)[0]
        cell = dict(id=rid, pts=(c + [x0, y0]).tolist(), area=int(area_of[rid]), nseeds=int(len(mine)),
                    pole=[int(pole[1]), int(pole[0])])
        if len(mine):
            # a plot number sits near the middle of its field: take the blob closest to the pole
            j = min(mine, key=lambda i: (seeds[i]["x"] - pole[1]) ** 2 + (seeds[i]["y"] - pole[0]) ** 2)
            cell["seed"] = [round(seeds[j]["x"], 1), round(seeds[j]["y"], 1)]
            cell["seed_bbox"] = seeds[j]["bbox"]
        cells.append(cell)
    return cells

def keep_main_tiling(cells, W, H):
    """Cells that form one connected mass = the parish. Islands (title box, scale bar, signature) are dropped."""
    if not cells: return cells
    m = np.zeros((H, W), np.uint8)
    for c in cells: cv2.fillPoly(m, [np.array(c["pts"], np.int32)], 255)
    m = cv2.dilate(m, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return [c for c in cells if lab[min(max(c["pole"][1], 0), H - 1), min(max(c["pole"][0], 0), W - 1)] == big]

def preview(img, cells, path):
    out = img.copy()
    ov = img.copy()
    for c in cells:
        col = (60, 200, 60) if c["nseeds"] == 1 else (0, 140, 255) if c["nseeds"] > 1 else (170, 170, 170)
        cv2.fillPoly(ov, [np.array(c["pts"], np.int32)], col)
    out = cv2.addWeighted(img, 0.6, ov, 0.4, 0)
    for c in cells:
        cv2.polylines(out, [np.array(c["pts"], np.int32)], True, (90, 30, 0), 1, cv2.LINE_AA)
        if "seed" in c: cv2.circle(out, (int(c["seed"][0]), int(c["seed"][1])), 3, (0, 0, 255), -1)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 88])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", default="auto_cells.json")
    ap.add_argument("--preview", default="auto_preview.jpg")
    ap.add_argument("--lo", type=float, default=P["lo"]); ap.add_argument("--hi", type=float, default=P["hi"])
    a = ap.parse_args()
    P["lo"], P["hi"] = a.lo, a.hi
    cv2.setNumThreads(1)
    img0 = cv2.imread(a.image); assert img0 is not None, f"cannot read {a.image}"
    img, reg, seeds, Ds = build(img0, a.scale)
    H, W = img.shape[:2]
    cells = keep_main_tiling(polygons(reg, seeds, Ds, W, H), W, H)
    log(f"{len(cells)} cells: {sum(c['nseeds']==1 for c in cells)} with exactly one number blob, "
        f"{sum(c['nseeds']==0 for c in cells)} with none, {sum(c['nseeds']>1 for c in cells)} with several")
    json.dump(dict(scale=a.scale, size=[W, H], cells=cells), open(a.out, "w"))
    preview(img, cells, a.preview)
    log("wrote", a.out, a.preview)

if __name__ == "__main__":
    main()
