"""Minimal aligned-sample generator for the NSF-FMRG DED data challenge.

One sample = a thermal tensor (frames near a physical x) + a geometry-based track-width label.
The width is extracted from the actual Wyko height values: a local cross-track profile
is detrended against the outer substrate, the deposited feature is segmented by its
signed height excursion, and its left/right boundaries define the local width.

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
from scipy.ndimage import binary_closing, gaussian_filter1d, median_filter

from nsf_fmrg_data import extract_final_thermal_frames, load_wyko_asc

TRACK_IDS = (8, 10, 14, 21)
GEOMETRY_LABEL_VERSION = "ridge-boundary-v4-stable-profile"


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


def _fill_nan_linear(values):
    """Linearly fill internal NaNs for 1-D smoothing; keep all-NaN profiles invalid."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return None
    coordinates = np.arange(values.size)
    return np.interp(coordinates, coordinates[valid], values[valid])


def _robust_line(x, y, iterations=3):
    """Fit a baseline line while rejecting large residuals from the deposited feature."""
    valid = np.isfinite(y)
    if valid.sum() < 4:
        return None
    keep = valid.copy()
    coefficients = None
    for _ in range(iterations):
        if keep.sum() < 4:
            break
        coefficients = np.polyfit(x[keep], y[keep], deg=1)
        residual = y - np.polyval(coefficients, x)
        scale = 1.4826 * np.nanmedian(np.abs(residual[keep] - np.nanmedian(residual[keep])))
        if not np.isfinite(scale) or scale <= 1e-12:
            break
        keep = valid & (np.abs(residual) <= 2.5 * scale)
    return coefficients


def _interpolate_threshold_crossing(y, signal, threshold, left_index, right_index):
    """Return the sub-pixel crossing of signal(y) with threshold, or NaN if invalid."""
    s0, s1 = signal[left_index], signal[right_index]
    if not np.isfinite(s0) or not np.isfinite(s1) or s0 == s1:
        return np.nan
    fraction = (threshold - s0) / (s1 - s0)
    if not 0.0 <= fraction <= 1.0:
        return np.nan
    return float(y[left_index] + fraction * (y[right_index] - y[left_index]))


def _largest_false_run(mask):
    """Length of the longest False run in a Boolean mask."""
    run_length, _, _ = _longest_run(~np.asarray(mask, dtype=bool))
    return run_length


def _apply_label_quality_gates(track_id, x_mm, width_mm, quality, confidence, reasons,
                               minimum_confidence=0.70,
                               continuity_tolerance_fraction=0.25,
                               minimum_continuity_tolerance_um=50.0,
                               neighbour_count=5):
    """Reject weak or positionally isolated width labels before ML.

    A true deposited-track width should evolve continuously over adjacent 0.2-mm
    scan positions.  This is deliberately a *label QA* gate, rather than a model
    feature: it prevents a short segmentation artifact from becoming a supervised
    target.  Its tolerance scales with local width, with a conservative micron floor.
    """
    accepted = np.asarray(quality, dtype=bool).copy()
    reasons = np.asarray(reasons, dtype=str).copy()
    low_confidence = accepted & (np.asarray(confidence) < minimum_confidence)
    accepted[low_confidence] = False
    reasons[low_confidence] = "low_confidence"

    # Compare each candidate with the median of its nearest accepted neighbours
    # from the *same track*.  Use the pre-continuity set for all comparisons so
    # filtering order cannot cascade through a track.
    pre_continuity = accepted.copy()
    for track in np.unique(track_id):
        index = np.flatnonzero((track_id == track) & pre_continuity)
        if index.size < 2 * neighbour_count:
            continue
        widths_um = width_mm[index] * 1000.0
        for position, source_index in enumerate(index):
            lo = max(0, position - neighbour_count)
            hi = min(index.size, position + neighbour_count + 1)
            neighbour_widths = np.delete(widths_um[lo:hi], position - lo)
            if neighbour_widths.size < neighbour_count:
                continue
            local_median = np.median(neighbour_widths)
            tolerance_um = max(
                minimum_continuity_tolerance_um,
                continuity_tolerance_fraction * max(local_median, 50.0),
            )
            if abs(widths_um[position] - local_median) > tolerance_um:
                accepted[source_index] = False
                reasons[source_index] = "x_discontinuous_width"
    return accepted, reasons


def _smooth_accepted_widths(x_mm, width_mm, quality, window=5):
    """Median-smooth accepted labels within contiguous x runs, retaining raw widths too."""
    smoothed = np.full_like(width_mm, np.nan, dtype=float)
    accepted = np.flatnonzero(quality)
    if accepted.size == 0:
        return smoothed
    step = np.median(np.diff(x_mm)) if x_mm.size > 1 else np.inf
    split = np.flatnonzero(np.diff(x_mm[accepted]) > 1.5 * step) + 1
    for run in np.split(accepted, split):
        if run.size < window:
            smoothed[run] = width_mm[run]
        else:
            smoothed[run] = median_filter(width_mm[run], size=window, mode="nearest")
    return smoothed


class SampleGenerator:
    """Lazily loads each track's thermal + height once, then serves samples."""

    def __init__(self, project_dir):
        root = Path(project_dir)
        self.thermal_dir = root / 'data' / 'raw' / 'thermal'
        self.height_dir = root / 'data' / 'raw' / 'height_maps'
        self._thermal = {}
        self._height = {}
        self._track_signal_level = {}

    # ---- raw loaders (cached per track) ----
    def thermal(self, track):
        if track not in self._thermal:
            r = extract_final_thermal_frames(self.thermal_dir, track)
            self._thermal[track] = (np.ascontiguousarray(r['frames']), np.asarray(r['x_mm_center']))
        return self._thermal[track]                      # frames (N,400,400), x_mm_center (N,)

    def height(self, track):
        if track not in self._height:
            r = load_wyko_asc(self.height_dir, track)
            self._height[track] = (r['Z_mm'], r['x_actual_mm'], r['y_mm'])
        return self._height[track]                       # Z_mm (M,Nx), x_mm (Nx,), y_mm (M,)

    def _estimate_track_signal_level(self, track, lx=75, sample_stride=100, polarity=1):
        """Estimate and cache a stable, track-wide deposited-feature signal level."""
        cache_key = (int(track), int(lx), int(sample_stride), int(polarity))
        if cache_key in self._track_signal_level:
            return self._track_signal_level[cache_key]
        z_mm, x_mm, y_mm = self.height(track)
        outer_count = max(8, int(round(0.20 * z_mm.shape[0])))
        outer_indices = np.r_[np.arange(outer_count), np.arange(z_mm.shape[0] - outer_count, z_mm.shape[0])]
        estimates = []
        for center in range(0, z_mm.shape[1], sample_stride):
            x0, x1 = max(0, center - lx // 2), min(z_mm.shape[1], center + lx // 2 + 1)
            profile_mm = np.nanmedian(z_mm[:, x0:x1], axis=1)
            filled = _fill_nan_linear(profile_mm)
            if filled is None:
                continue
            coefficients = _robust_line(y_mm[outer_indices], profile_mm[outer_indices])
            if coefficients is None:
                continue
            residual_um = (filled - np.polyval(coefficients, y_mm)) * 1000.0
            estimates.append(float(np.nanpercentile(polarity * gaussian_filter1d(residual_um, sigma=2.0), 95)))
        level = float(np.nanmedian(estimates)) if estimates else np.nan
        self._track_signal_level[cache_key] = level
        return level

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

    def width_at(
        self,
        track,
        x,
        lx=75,
        smooth_sigma_px=2.0,
        minimum_height_um=3.0,
        minimum_width_px=8,
        maximum_width_fraction=0.80,
        maximum_internal_gap_px=2,
        track_polarity=1,
    ):
        """Extract local deposited-track width from the actual height-map geometry.

        A local median profile over an x-window is detrended with a robust line fitted to
        valid substrate flanks. A calibrated ridge/depression polarity is segmented;
        short gaps are bridged, but boundary crossings are refined against the detrended
        residual by linear interpolation. Returned ``quality`` rejects profiles without
        valid flanks, weak/edge-touching/broad features, or bridged missing-data gaps.
        """
        z_mm, x_mm, y_mm = self.height(track)
        pixel_mm = float(np.median(np.diff(y_mm)))
        rows = z_mm.shape[0]
        center = int(np.argmin(np.abs(x_mm - x)))
        x0, x1 = max(0, center - lx // 2), min(z_mm.shape[1], center + lx // 2 + 1)
        profile_mm = np.nanmedian(z_mm[:, x0:x1], axis=1)
        finite_fraction = np.isfinite(z_mm[:, x0:x1]).mean(axis=1)
        filled_profile = _fill_nan_linear(profile_mm)

        result = {
            'width_mm': np.nan, 'left_boundary_mm': np.nan, 'right_boundary_mm': np.nan,
            'baseline_mm': np.nan, 'peak_height_um': np.nan, 'threshold_um': np.nan,
            'polarity': 0, 'quality': False, 'confidence': 0.0,
            'rejection_reason': 'insufficient_profile_data',
            'profile_mm': profile_mm, 'residual_um': np.full(rows, np.nan),
            'smoothed_residual_um': np.full(rows, np.nan),
            'baseline_profile_mm': np.full(rows, np.nan),
            'feature_mask': np.zeros(rows, dtype=bool), 'y_mm': y_mm,
        }
        if filled_profile is None or x < x_mm[0] or x > x_mm[-1]:
            return result

        outer_count = max(8, int(round(0.20 * rows)))
        outer_indices = np.r_[np.arange(outer_count), np.arange(rows - outer_count, rows)]
        coefficients = _robust_line(y_mm[outer_indices], profile_mm[outer_indices])
        if coefficients is None:
            result['rejection_reason'] = 'insufficient_valid_substrate_flanks'
            return result

        baseline_mm = np.polyval(coefficients, y_mm)
        residual_um = (filled_profile - baseline_mm) * 1000.0
        flank_valid = np.isfinite(profile_mm[outer_indices])
        outer_residual = residual_um[outer_indices][flank_valid]
        if outer_residual.size < 8:
            result['rejection_reason'] = 'insufficient_valid_substrate_flanks'
            return result
        noise_um = 1.4826 * np.median(np.abs(outer_residual - np.median(outer_residual)))
        smooth_residual_um = gaussian_filter1d(residual_um, sigma=smooth_sigma_px, mode='nearest')

        if track_polarity not in (-1, 1):
            raise ValueError("track_polarity must be +1 for a ridge or -1 for a depression.")
        # QA across all four tracks shows the deposited feature is the broad positive
        # ridge. Narrow negative excursions are interferometry dropouts, not track edges.
        polarity = int(track_polarity)
        signed_signal_um = polarity * smooth_residual_um
        signal_level_um = self._estimate_track_signal_level(track, lx=lx, polarity=polarity)
        if not np.isfinite(signal_level_um):
            signal_level_um = float(np.nanpercentile(signed_signal_um, 95))
        threshold_um = max(
            float(minimum_height_um),
            2.0 * float(noise_um),
            0.45 * signal_level_um,
        )
        candidate = np.isfinite(profile_mm) & (signed_signal_um >= threshold_um)
        candidate = binary_closing(
            candidate,
            structure=np.ones(int(maximum_internal_gap_px) + 1, dtype=bool),
        )
        run_length, start, stop = _longest_run(candidate)
        feature_mask = np.zeros(rows, dtype=bool)
        if run_length:
            feature_mask[start:stop] = True

        too_wide = run_length > int(maximum_width_fraction * rows)
        finite_support = float(np.mean(finite_fraction[start:stop])) if run_length else 0.0
        internal_valid = np.isfinite(profile_mm[start:stop]) if run_length else np.array([], dtype=bool)
        largest_gap = _largest_false_run(internal_valid) if run_length else 0
        peak_um = float(np.nanmax(signed_signal_um))
        left_refined = (
            _interpolate_threshold_crossing(y_mm, signed_signal_um, threshold_um, start - 1, start)
            if start > 0 else np.nan
        )
        right_refined = (
            _interpolate_threshold_crossing(y_mm, signed_signal_um, threshold_um, stop - 1, stop)
            if stop < rows else np.nan
        )
        rejection_reasons = []
        if run_length < minimum_width_px: rejection_reasons.append('feature_too_narrow')
        if start == 0 or stop == rows: rejection_reasons.append('feature_touches_y_boundary')
        if too_wide: rejection_reasons.append('feature_too_broad')
        if finite_support < 0.50: rejection_reasons.append('insufficient_feature_support')
        if largest_gap > maximum_internal_gap_px: rejection_reasons.append('internal_missing_data_gap')
        if peak_um < threshold_um: rejection_reasons.append('below_height_threshold')
        if not np.isfinite(left_refined) or not np.isfinite(right_refined): rejection_reasons.append('unrefinable_boundary')
        quality = bool(
            not rejection_reasons
        )
        confidence = (
            min(1.0, peak_um / max(2.0 * threshold_um, 1e-12))
            * finite_support
            * min(1.0, run_length / max(2.0 * minimum_width_px, 1.0))
        )

        result.update({
            'width_mm': abs(right_refined - left_refined) if quality else np.nan,
            'left_boundary_mm': left_refined if quality else np.nan,
            'right_boundary_mm': right_refined if quality else np.nan,
            'baseline_mm': float(np.median(baseline_mm)),
            'peak_height_um': peak_um,
            'threshold_um': threshold_um,
            'polarity': polarity,
            'quality': quality,
            'confidence': confidence if quality else 0.0,
            'rejection_reason': 'accepted' if quality else ';'.join(rejection_reasons),
            'residual_um': residual_um,
            'smoothed_residual_um': smooth_residual_um,
            'baseline_profile_mm': baseline_mm,
            'feature_mask': feature_mask,
        })
        return result

    # ---- one sample ----
    def make_sample(self, track_id, x, k=2, lx=75):
        wd = self.width_at(track_id, x, lx=lx)
        return {'track_id': track_id, 'x': float(x),
                'thermal': self.thermal_tensor(track_id, x, k),
                'width_mm': wd['width_mm'], 'quality': wd['quality']}

    # ---- batch: cheap width labels over the frame grid (no thermal) ----
    def build_labels(
        self,
        tracks=TRACK_IDS,
        lx=75,
        trim=1,
        minimum_confidence=0.70,
        continuity_tolerance_fraction=0.25,
    ):
        """Geometry labels for every frame centre, with final confidence/continuity QA."""
        xs = self.frame_grid()
        if trim:
            xs = xs[trim:len(xs) - trim]
        tid, xa, wa, qa, la, ra, pa, ca, tha, reasons = [], [], [], [], [], [], [], [], [], []
        for tr in tracks:
            for x in xs:
                wd = self.width_at(tr, float(x), lx=lx)
                tid.append(tr); xa.append(float(x)); wa.append(wd['width_mm']); qa.append(wd['quality'])
                la.append(wd['left_boundary_mm']); ra.append(wd['right_boundary_mm'])
                pa.append(wd['polarity']); ca.append(wd['confidence']); tha.append(wd['peak_height_um'])
                reasons.append(wd['rejection_reason'])
        tid = np.array(tid)
        xa = np.array(xa)
        wa = np.array(wa)
        raw_quality = np.array(qa, dtype=bool)
        ca = np.array(ca)
        final_quality, reasons = _apply_label_quality_gates(
            tid, xa, wa, raw_quality, ca, reasons,
            minimum_confidence=minimum_confidence,
            continuity_tolerance_fraction=continuity_tolerance_fraction,
        )
        # A rejected label must never carry a numeric target into downstream code.
        raw_width = wa.copy()
        wa[~final_quality] = np.nan
        la = np.asarray(la); la[~final_quality] = np.nan
        ra = np.asarray(ra); ra[~final_quality] = np.nan
        model_width = np.full_like(wa, np.nan, dtype=float)
        for track in np.unique(tid):
            track_mask = tid == track
            model_width[track_mask] = _smooth_accepted_widths(
                xa[track_mask], wa[track_mask], final_quality[track_mask]
            )
        return {'track_id': tid, 'x_mm': xa,
                'width_mm': wa, 'raw_width_mm': raw_width, 'model_width_mm': model_width,
                'quality': final_quality, 'raw_geometry_quality': raw_quality,
                'left_boundary_mm': la, 'right_boundary_mm': ra,
                'polarity': np.array(pa, dtype=np.int8), 'confidence': np.array(ca),
                'peak_height_um': np.array(tha), 'rejection_reason': np.array(reasons)}
