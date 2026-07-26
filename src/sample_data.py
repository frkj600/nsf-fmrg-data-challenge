"""Minimal aligned-sample generator for the NSF-FMRG DED data challenge.

One sample = a thermal tensor (frames near a physical x) + a remelt-track width label
(2D box-averaged profilometer validity, thresholded, longest cross-track run).

    from sample_data import SampleGenerator
    gen = SampleGenerator('/path/to/nsf-fmrg-data-challenge')
    s = gen.make_sample(track_id=8, x=60.0)          # {track_id, x, thermal, width_mm, quality}
    labels = gen.build_labels()                       # width+quality over the 400-frame grid, all tracks

Reads raw data through the challenge's own loaders in nsf_fmrg_data.py (no cache dependency).
Thermal tensors are ~3 MB each (2k+1, 400, 400), so fetch them on demand rather than stacking
the whole grid (~5 GB); build_labels() returns only the cheap width labels + metadata.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from nsf_fmrg_data import extract_final_thermal_frames, load_wyko_asc

TRACK_IDS = (8, 10, 14, 21)


def _longest_run(mask):
    """(length, start, stop) of the longest True run in a 1-D bool array; (0,0,0) if none."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0, 0, 0
    br = np.where(np.diff(idx) > 1)[0]
    s = np.r_[idx[0], idx[br + 1]]
    e = np.r_[idx[br] + 1, idx[-1] + 1]
    j = int(np.argmax(e - s))
    return int(e[j] - s[j]), int(s[j]), int(e[j])


class SampleGenerator:
    """Lazily loads each track's thermal + height once, then serves samples."""

    def __init__(self, project_dir):
        root = Path(project_dir)
        self.thermal_dir = root / 'data' / 'raw' / 'thermal'
        self.height_dir = root / 'data' / 'raw' / 'height_maps'
        self._thermal = {}
        self._height = {}

    # ---- raw loaders (cached per track) ----
    def thermal(self, track):
        if track not in self._thermal:
            r = extract_final_thermal_frames(self.thermal_dir, track)
            self._thermal[track] = (np.ascontiguousarray(r['frames']), np.asarray(r['x_mm_center']))
        return self._thermal[track]                      # frames (N,400,400), x_mm_center (N,)

    def height(self, track):
        if track not in self._height:
            r = load_wyko_asc(self.height_dir, track)
            self._height[track] = (np.isfinite(r['Z_mm']), r['x_actual_mm'], r['y_mm'])
        return self._height[track]                       # valid (M,Nx), x_mm (Nx,), y_mm (M,)

    # ---- the physical x grid ----
    def frame_grid(self, track=8):
        """The 400 thermal frame centres in mm (20.1 .. 99.9); identical across tracks."""
        return self.thermal(track)[1]

    # ---- the two modalities ----
    def thermal_tensor(self, track, x, k=2):
        """(2k+1, 400, 400): the frame nearest physical x, plus k neighbours each side (clipped)."""
        frames, xc = self.thermal(track)
        c = int(np.argmin(np.abs(xc - x)))
        idx = np.clip(np.arange(c - k, c + k + 1), 0, len(xc) - 1)
        return frames[idx]

    def width_at(self, track, x, lx=25, ly=10, p=0.5):
        """Width (mm) = longest cross-track run whose lx-by-ly box-averaged validity exceeds p.

        Returns a dict with width_mm, the band (top/bot mm), the density profile, and a `quality`
        flag: True only if a run was found, it does not touch the cross-track FOV edges, and x lies
        within this track's height coverage (else argmin clamps to a far column)."""
        valid, xm, ym = self.height(track)
        pix = float(ym[1] - ym[0])
        m = valid.shape[0]
        j = int(np.argmin(np.abs(xm - x)))
        j0, j1 = max(0, j - lx // 2), min(valid.shape[1], j + lx // 2 + 1)
        rho = uniform_filter1d(valid[:, j0:j1].mean(axis=1).astype(np.float32), ly)  # lx (x) then ly (y)
        runlen, s, e = _longest_run(rho > p)
        quality = bool(runlen > 0 and s > 0 and e < m and xm[0] <= x <= xm[-1])
        return {'width_mm': runlen * pix,
                'top_mm': float(ym[s]) if runlen else np.nan,
                'bot_mm': float(ym[e - 1]) if runlen else np.nan,
                'rho': rho, 'y_mm': ym, 'quality': quality}

    # ---- one sample ----
    def make_sample(self, track_id, x, k=2, lx=25, ly=10, p=0.5):
        wd = self.width_at(track_id, x, lx, ly, p)
        return {'track_id': track_id, 'x': float(x),
                'thermal': self.thermal_tensor(track_id, x, k),
                'width_mm': wd['width_mm'], 'quality': wd['quality']}

    # ---- batch: cheap width labels over the frame grid (no thermal) ----
    def build_labels(self, tracks=TRACK_IDS, lx=25, ly=10, p=0.5, trim=1):
        """width_mm + quality for every (track, frame-centre x); trims `trim` frames off each end."""
        xs = self.frame_grid()
        if trim:
            xs = xs[trim:len(xs) - trim]
        tid, xa, wa, qa = [], [], [], []
        for tr in tracks:
            for x in xs:
                wd = self.width_at(tr, float(x), lx, ly, p)
                tid.append(tr); xa.append(float(x)); wa.append(wd['width_mm']); qa.append(wd['quality'])
        return {'track_id': np.array(tid), 'x_mm': np.array(xa),
                'width_mm': np.array(wa), 'quality': np.array(qa, dtype=bool)}
