"""
PPG Signal Processing Pipeline
==============================
Transforms raw PPG into heartbeat-level representation for reliable HR/HRV analysis.

Pipeline:
1. Artifact rejection - mask bad segments (DC jumps, saturation, motion)
2. DC removal - eliminates baseline (finger pressure, tissue, ambient light)
3. Band-pass filter (0.5-5 Hz) - suppresses motion artifacts and high-freq noise
4. Peak detection - identifies individual pulse beats
5. Compute HR, HRV (RMSSD, SDNN) from inter-beat intervals
"""

import numpy as np
from scipy import signal as scipy_signal


# Valid IBI range: 40-120 BPM = 500-1500 ms
IBI_MIN_MS = 400   # 150 BPM max
IBI_MAX_MS = 1500  # 40 BPM min

# 3-part raw PPG at 100 Hz - use for beat detection. 10-part vitals at 10 Hz.
FS_DEFAULT = 100.0


def mask_artifacts(signal: np.ndarray, grad_thresh_ratio: float = 0.15) -> np.ndarray:
    """
    Identify artifact segments: large DC jumps, saturation, motion.
    Returns boolean mask: True = good, False = artifact.
    """
    if len(signal) < 10:
        return np.ones(len(signal), dtype=bool)
    
    sig = np.array(signal, dtype=float)
    mask = np.ones(len(sig), dtype=bool)
    
    # 1. Large gradients = DC jump or motion artifact
    grad = np.abs(np.diff(sig))
    thresh = np.percentile(grad, 95) * 3  # Allow some variation
    if thresh < 1000:
        thresh = 1000
    bad_diff = np.where(grad > thresh)[0]
    for i in bad_diff:
        # Mask 0.5 sec around artifact
        w = max(5, len(sig) // 50)
        mask[max(0, i - w) : min(len(sig), i + w + 1)] = False
    
    # 2. Too flat = saturation or no finger (pulsatile component absent)
    win = min(50, len(sig) // 4)
    if win >= 10:
        thresh = max(1000, np.median(sig) * 0.005)
        for i in range(0, len(sig) - win, win // 2):
            seg = sig[i : i + win]
            if np.ptp(seg) < thresh:
                mask[i : i + win] = False
    
    # 3. Very low signal = no finger
    if np.median(sig) < 10000:
        mask[:] = False
    
    return mask


def extract_best_segment(signal: np.ndarray, mask: np.ndarray, min_len: int = 100):
    """
    Extract the longest contiguous good segment.
    Returns (segment, start_idx) or (empty, 0) if none found.
    Prefer segment at the END (most recent data).
    """
    if np.sum(mask) < min_len:
        return np.array([]), 0
    
    padded = np.concatenate([[0], mask.astype(int), [0]])
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    best_start, best_end = 0, 0
    for s, e in zip(starts, ends):
        if e - s >= min_len and e - s > best_end - best_start:
            best_start, best_end = s, e
        elif e - s >= min_len and e - s == best_end - best_start and e > best_end:
            best_start, best_end = s, e
    
    if best_end - best_start >= min_len:
        return signal[best_start:best_end], best_start
    return np.array([]), 0


def remove_dc(signal: np.ndarray, method: str = "ema") -> np.ndarray:
    """
    Remove DC component (baseline) from the PPG signal.
    EMA tracks baseline drift better than mean for signals with DC jumps.
    """
    if method == "mean":
        return signal - np.mean(signal)
    elif method == "ema":
        alpha = 0.02  # Baseline tracking - faster than 0.01 for drift
        baseline = np.zeros_like(signal, dtype=float)
        baseline[0] = signal[0]
        for i in range(1, len(signal)):
            baseline[i] = alpha * signal[i] + (1 - alpha) * baseline[i - 1]
        return signal - baseline
    else:
        return signal - np.mean(signal)


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 5.0,
    fs: float = FS_DEFAULT,
    order: int = 4
) -> np.ndarray:
    """
    Band-pass filter to preserve physiological heart-rate range (0.5-5 Hz).
    Suppresses motion artifacts and high-frequency noise.
    
    Args:
        signal: DC-removed PPG
        lowcut: Low cutoff Hz (removes baseline drift)
        highcut: High cutoff Hz (removes noise)
        fs: Sample rate (Hz)
        order: Butterworth filter order
    
    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    highcut_safe = min(highcut, nyq * 0.99)
    low = lowcut / nyq
    high = highcut_safe / nyq
    low = min(low, 0.99)
    high = min(high, 0.99)
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    x = signal.astype(float)
    # filtfilt needs len(x) > padlen; default padlen ~27 for order-4 -> requires 28+ samples
    padlen = min(3 * max(len(a), len(b)), len(x) - 2)
    padlen = max(1, padlen)
    return scipy_signal.filtfilt(b, a, x, padlen=padlen)


def detect_peaks(
    signal: np.ndarray,
    fs: float = FS_DEFAULT,
    min_distance_samples: int = None,
    prominence_ratio: float = 0.05,
) -> np.ndarray:
    """Detect systolic peaks in PPG. PPG peaks are rounded - use relaxed params."""
    if len(signal) < 10:
        return np.array([])

    if min_distance_samples is None:
        min_distance_samples = max(1, int(0.4 * fs))  # 400 ms = 150 BPM max

    sig_range = np.ptp(signal)
    if sig_range < 1e-6:
        return np.array([])

    prominence = max(sig_range * prominence_ratio, 1.0)
    height_thresh = np.median(signal) + 0.05 * sig_range

    peaks, _ = scipy_signal.find_peaks(
        signal,
        distance=min_distance_samples,
        prominence=prominence,
        height=height_thresh,
    )
    return peaks


def peaks_to_ibi(peak_indices: np.ndarray, fs: float = FS_DEFAULT) -> np.ndarray:
    """
    Compute inter-beat intervals (IBI) in milliseconds from peak indices.
    
    Args:
        peak_indices: Sample indices of detected peaks
        fs: Sample rate (Hz)
    
    Returns:
        Array of IBI in ms (one less than number of peaks)
    """
    if len(peak_indices) < 2:
        return np.array([])
    intervals_samples = np.diff(peak_indices)
    return intervals_samples * (1000.0 / fs)


def filter_valid_ibi(ibi_ms: np.ndarray) -> np.ndarray:
    """Keep only physiologically valid IBIs (40-120 BPM)."""
    valid = ibi_ms[(ibi_ms >= IBI_MIN_MS) & (ibi_ms <= IBI_MAX_MS)]
    if len(valid) < 3:
        return valid
    # Reject outliers: IBI shouldn't jump > 300ms beat-to-beat
    keep = np.ones(len(valid), dtype=bool)
    for i in range(1, len(valid)):
        if np.abs(valid[i] - valid[i - 1]) > 300:
            keep[i] = False
    return valid[keep]


def compute_hr(ibi_ms: np.ndarray) -> float:
    """Heart rate from valid IBIs (BPM). Returns 0 if insufficient data."""
    valid = filter_valid_ibi(ibi_ms)
    if len(valid) < 1:
        return 0.0
    return 60000.0 / np.median(valid)


def compute_hrv_rmssd(ibi_ms: np.ndarray) -> float:
    """
    HRV as RMSSD (Root Mean Square of Successive Differences).
    Reflects parasympathetic activity.
    """
    valid = filter_valid_ibi(ibi_ms)
    if len(valid) < 2:
        return 0.0
    diffs = np.diff(valid)
    return np.sqrt(np.mean(diffs ** 2))


def compute_hrv_sdnn(ibi_ms: np.ndarray) -> float:
    """
    HRV as SDNN (Standard Deviation of NN intervals).
    Reflects overall variability.
    """
    valid = filter_valid_ibi(ibi_ms)
    if len(valid) < 2:
        return 0.0
    return np.std(valid)


def process_ppg_pipeline(
    signal: np.ndarray,
    fs: float = FS_DEFAULT,
    timestamps_ms: np.ndarray = None,
) -> dict:
    """
    Full pipeline: artifact mask -> best segment -> DC removal -> band-pass -> peak detection -> HR/HRV.
    """
    if len(signal) < 30:
        return {
            "peaks": np.array([]),
            "ibi_ms": np.array([]),
            "hr": 0.0,
            "hrv_rmssd": 0.0,
            "hrv_sdnn": 0.0,
            "filtered_signal": np.zeros(len(signal)),
            "quality": 0.0,
        }
    
    # Use full signal. Artifact masking (mask_artifacts) is implemented but disabled for demo
    # stability under low sample rates (10 Hz). Enable for 100 Hz raw streams if needed.
    segment = signal
    seg_start = 0
    
    # 2. DC removal (EMA tracks baseline better)
    dc_removed = remove_dc(segment, method="ema")
    
    # 3. Band-pass filter
    filtered = bandpass_filter(dc_removed, fs=fs)
    
    # 4. Peak detection (stricter)
    peaks = detect_peaks(filtered, fs=fs)
    
    # 5. Compute IBIs
    ibi_ms = peaks_to_ibi(peaks, fs=fs)
    
    # 6. HR and HRV
    hr = compute_hr(ibi_ms)
    hrv_rmssd = compute_hrv_rmssd(ibi_ms)
    hrv_sdnn = compute_hrv_sdnn(ibi_ms)
    
    # Quality: need enough valid beats, reasonable variance
    valid_ibi = filter_valid_ibi(ibi_ms)
    quality = min(100, len(valid_ibi) * 15) if len(valid_ibi) >= 2 else 0
    
    # Place filtered at correct position for plotting
    filt_padded = np.zeros(len(signal))
    filt_padded[seg_start : seg_start + len(filtered)] = filtered
    peaks_full = (seg_start + peaks) if len(peaks) > 0 else np.array([])
    
    return {
        "peaks": peaks_full,
        "ibi_ms": ibi_ms,
        "hr": hr,
        "hrv_rmssd": hrv_rmssd,
        "hrv_sdnn": hrv_sdnn,
        "filtered_signal": filt_padded,
        "quality": quality,
    }
