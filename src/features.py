# src/features.py

import numpy as np
from scipy import stats
from blimpy import Waterfall

# ─────────────────────────────────────────
# 1. FIND REGIONS OF INTEREST (ROIs)
# ─────────────────────────────────────────

def find_regions(snr, freqs, snr_threshold=10.0, min_gap_hz=0.01):
    """
    Scan the SNR array and group high-SNR channels into
    distinct "regions of interest" (ROIs) — i.e. candidate signals.

    Think of it like highlighting bright spots on the spectrogram:
    any channel above the threshold gets flagged, and neighboring
    flagged channels get merged into one region (one signal).

    Args:
        snr           : 1D array of SNR per frequency channel
        freqs         : 1D array of frequency values in MHz
        snr_threshold : minimum SNR to count as a signal (default 10)
        min_gap_hz    : minimum gap in MHz to treat two blobs as separate signals

    Returns:
        regions : list of dicts, each describing one detected signal
                  keys → 'freq_start', 'freq_stop', 'peak_freq',
                          'peak_snr', 'bandwidth_hz', 'channel_indices'
    """
    print(f"[1/3] Finding regions of interest (SNR threshold: {snr_threshold})...")

    # Boolean mask — True where SNR is above threshold
    hot = snr > snr_threshold

    regions    = []
    in_region  = False
    start_idx  = 0

    for i in range(len(hot)):
        if hot[i] and not in_region:
            # entered a new hot region
            in_region = True
            start_idx = i

        elif not hot[i] and in_region:
            # just left a hot region — save it
            end_idx = i - 1
            in_region = False

            region_snr   = snr[start_idx:end_idx + 1]
            region_freqs = freqs[start_idx:end_idx + 1]
            peak_idx     = np.argmax(region_snr)

            regions.append({
                "freq_start"      : float(region_freqs.min()),
                "freq_stop"       : float(region_freqs.max()),
                "peak_freq"       : float(region_freqs[peak_idx]),
                "peak_snr"        : float(region_snr[peak_idx]),
                "bandwidth_hz"    : float((region_freqs.max() - region_freqs.min()) * 1e6),
                "channel_indices" : list(range(start_idx, end_idx + 1)),
            })

    # catch a region that runs to the end of the array
    if in_region:
        region_snr   = snr[start_idx:]
        region_freqs = freqs[start_idx:]
        peak_idx     = np.argmax(region_snr)
        regions.append({
            "freq_start"      : float(region_freqs.min()),
            "freq_stop"       : float(region_freqs.max()),
            "peak_freq"       : float(region_freqs[peak_idx]),
            "peak_snr"        : float(region_snr[peak_idx]),
            "bandwidth_hz"    : float((region_freqs.max() - region_freqs.min()) * 1e6),
            "channel_indices" : list(range(start_idx, len(snr))),
        })

    print(f"    Found {len(regions)} region(s) above SNR {snr_threshold}")
    return regions


# ─────────────────────────────────────────
# 2. CALCULATE DRIFT RATE
# ─────────────────────────────────────────

def calculate_drift_rate(data, freqs, channel_indices, tsamp=18.25):
    """
    Estimate the drift rate (Hz/sec) of a signal by tracking
    where its peak frequency sits at each time step.

    Method: Linear regression on peak_frequency vs time.
            The slope = drift rate in Hz/sec.

    A non-zero drift rate is one of the strongest indicators of
    a real extraterrestrial signal (due to Doppler shift from
    relative motion between source and Earth).

    RFI from ground/satellites typically has near-zero drift,
    OR a very large erratic drift. ET signals should have a
    smooth, consistent drift matching orbital mechanics.

    Args:
        data             : numpy array (time_steps, 1, freq_channels)
        freqs            : 1D array of frequency values in MHz
        channel_indices  : list of channel indices belonging to this region
        tsamp            : time per integration step in seconds (18.25s for this file)

    Returns:
        drift_rate_hz_s  : float, drift rate in Hz per second
        r_squared        : float, quality of the linear fit (0–1)
                           closer to 1.0 = cleaner drift line
    """
    # Extract just the channels belonging to this region
    region_data = data[:, 0, channel_indices]   # shape: (time_steps, n_channels)
    region_freqs = freqs[channel_indices]        # MHz values for those channels

    time_steps = region_data.shape[0]
    times      = np.arange(time_steps) * tsamp  # convert step index → seconds

    # For each time step, find the frequency of the peak power channel
    peak_chan_per_step = np.argmax(region_data, axis=1)
    peak_freq_per_step = region_freqs[peak_chan_per_step]   # MHz at each time step

    # Linear regression: peak_freq ~ time
    # slope = how fast frequency is changing → drift rate
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        times,
        peak_freq_per_step * 1e6   # convert MHz → Hz for the result
    )

    r_squared = r_value ** 2

    return float(slope), float(r_squared)


# ─────────────────────────────────────────
# 3. EXTRACT ALL FEATURES
# ─────────────────────────────────────────

def extract_features(data, freqs, snr, snr_threshold=10.0, tsamp=18.25):
    """
    Master function: combines find_regions + calculate_drift_rate
    into one feature vector per detected signal.

    Each signal becomes a row with these columns:
        peak_freq_mhz  : where is it in the spectrum?
        bandwidth_hz   : how wide is it?
        peak_snr       : how strong is it?
        drift_rate     : is it moving in frequency? (Hz/sec)
        drift_r2       : how clean/linear is that drift? (0–1)

    These 5 numbers are what HDBSCAN will cluster in features.py.

    Args:
        data          : numpy array (time_steps, 1, freq_channels)
        freqs         : 1D array of frequency values in MHz
        snr           : 1D SNR array from calculate_snr()
        snr_threshold : cutoff for what counts as a signal
        tsamp         : seconds per time integration step

    Returns:
        feature_matrix : numpy array shape (n_signals, 5)
        feature_labels : list of column name strings
        regions        : raw region dicts (for debugging/plotting)
    """
    print("[Feature Extraction] Starting...")

    regions = find_regions(snr, freqs, snr_threshold=snr_threshold)

    if len(regions) == 0:
        print("    No regions found — try lowering snr_threshold")
        return np.array([]), [], []

    feature_rows = []

    print(f"[2/3] Extracting features from {len(regions)} region(s)...")

    for i, region in enumerate(regions):
        idx = region["channel_indices"]

        drift_rate, drift_r2 = calculate_drift_rate(
            data, freqs, idx, tsamp=tsamp
        )

        row = [
            region["peak_freq"],      # MHz  — where in spectrum
            region["bandwidth_hz"],   # Hz   — how wide
            region["peak_snr"],       # —    — how strong
            drift_rate,               # Hz/s — is it drifting?
            drift_r2,                 # —    — how linear is the drift?
        ]

        feature_rows.append(row)

        print(f"    Region {i+1:02d} | "
              f"freq={region['peak_freq']:.4f} MHz | "
              f"BW={region['bandwidth_hz']:.1f} Hz | "
              f"SNR={region['peak_snr']:.1f} | "
              f"drift={drift_rate:.4f} Hz/s | "
              f"R²={drift_r2:.3f}")

    feature_matrix = np.array(feature_rows)
    feature_labels = ["peak_freq_mhz", "bandwidth_hz",
                      "peak_snr", "drift_rate_hz_s", "drift_r2"]

    print(f"\n[3/3] Feature matrix shape: {feature_matrix.shape}")
    print(f"      Columns: {feature_labels}")

    return feature_matrix, feature_labels, regions


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    # import our preprocess functions
    import sys
    sys.path.insert(0, "src")
    from preprocess import load_data, integrate_time, calculate_snr

    FILE = "data/raw/Voyager1.single_coarse.fine_res.h5"

    # Run the full Phase 1 pipeline first
    wf, data, freqs         = load_data(FILE, f_start=8419.2, f_stop=8419.4)
    data_integrated         = integrate_time(data, n_steps=4)
    snr                     = calculate_snr(data_integrated)

    # Now extract features
    features, labels, regions = extract_features(
        data_integrated, freqs, snr,
        snr_threshold=10.0,
        tsamp=18.25 * 4        # tsamp × n_steps because we integrated
    )

    if features.size > 0:
        print("\n── Feature Matrix ──")
        print(f"{'':>3}  {'peak_freq':>12}  {'bandwidth':>10}  "
              f"{'snr':>8}  {'drift':>10}  {'R²':>6}")
        for i, row in enumerate(features):
            print(f"{i+1:>3}  {row[0]:>12.4f}  {row[1]:>10.1f}  "
                  f"{row[2]:>8.1f}  {row[3]:>10.4f}  {row[4]:>6.3f}")

    print("\n✅ features.py ran successfully!")