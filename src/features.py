# src/features.py

import numpy as np
from scipy import stats
from scipy import ndimage
from skimage import filters, measure
from blimpy import Waterfall


# ─────────────────────────────────────────
# 0a. DENOISE THE SPECTROGRAM
# ─────────────────────────────────────────

def denoise(data, median_filter_size=3):
    """
    Clean up the raw spectrogram before blob detection.

    Steps:
        1. Squeeze to 2D (time_steps, freq_channels)
        2. Apply a median filter to kill random spikes (salt & pepper noise)
        3. Subtract the noise floor so only structure remains

    Why median and not mean?
        A mean filter blurs everything including signal edges.
        A median filter kills isolated spikes while keeping
        the sharp edges of narrowband signals intact.

    Args:
        data             : numpy array (time_steps, 1, freq_channels)
        median_filter_size : size of the median filter kernel

    Returns:
        cleaned : 2D numpy array (time_steps, freq_channels), noise-subtracted
    """
    print("[0a] Denoising spectrogram...")

    # squeeze out the feed dimension → (time_steps, freq_channels)
    d = data[:, 0, :].astype(np.float32)

    # median filter — kills salt & pepper noise, preserves edges
    filtered = ndimage.median_filter(d, size=median_filter_size)

    # subtract the noise floor (median of each freq channel across time)
    # this "flattens" the bandpass so bright RFI doesn't drown faint signals
    noise_floor = np.median(filtered, axis=0)
    cleaned     = filtered - noise_floor

    # clip negatives to zero — we only care about things above the floor
    cleaned = np.clip(cleaned, 0, None)

    print(f"    Input shape  : {d.shape}")
    print(f"    Output shape : {cleaned.shape}")
    print(f"    Max value after cleaning: {cleaned.max():.2f}")

    return cleaned


# ─────────────────────────────────────────
# 0b. BLOB DETECTION — CONNECTED COMPONENTS
# ─────────────────────────────────────────

def detect_blobs(cleaned, threshold_sigma=3.0, min_area=5):
    """
    Find distinct signal "blobs" in the cleaned spectrogram.

    Method — Connected Component Labeling:
        1. Threshold: anything above (mean + N*sigma) becomes a 1, rest 0
           This creates a binary mask of "hot" pixels
        2. Label: scipy.ndimage finds connected groups of 1s and gives
           each group a unique integer label
        3. Measure: skimage.measure.regionprops extracts bounding box,
           area, centroid etc. for each labeled region
        4. Filter: drop tiny regions (likely noise spikes, not real signals)

    Args:
        cleaned         : 2D array from denoise() — (time_steps, freq_channels)
        threshold_sigma : how many std devs above mean = "hot pixel"
                          lower = more sensitive, higher = only bright signals
        min_area        : minimum pixel area to keep a blob (filters noise)

    Returns:
        blobs    : list of skimage RegionProps objects, one per detected blob
        label_map: 2D array same shape as cleaned, each pixel labeled by blob ID
                   (0 = background, 1,2,3... = blob IDs)
    """
    print(f"[0b] Detecting blobs (threshold: mean + {threshold_sigma}σ, "
          f"min area: {min_area} px)...")

    # ── Step 1: Threshold ──────────────────────────────────────────────────
    mean  = cleaned.mean()
    sigma = cleaned.std()
    threshold = mean + threshold_sigma * sigma

    binary = (cleaned > threshold).astype(np.uint8)

    print(f"    Noise floor   : mean={mean:.2f}, σ={sigma:.2f}")
    print(f"    Threshold     : {threshold:.2f}")
    print(f"    Hot pixels    : {binary.sum()} / {binary.size}")

    # ── Step 2: Label connected components ────────────────────────────────
    # structure defines connectivity — here we use full 8-connectivity
    # (diagonals count as connected, not just up/down/left/right)
    structure = ndimage.generate_binary_structure(2, 2)
    label_map, n_labels = ndimage.label(binary, structure=structure)

    print(f"    Raw blobs found: {n_labels}")

    # ── Step 3: Measure region properties ─────────────────────────────────
    blobs = measure.regionprops(label_map, intensity_image=cleaned)

    # ── Step 4: Filter small blobs ─────────────────────────────────────────
    blobs = [b for b in blobs if b.area >= min_area]

    print(f"    Blobs after size filter (>={min_area}px): {len(blobs)}")

    return blobs, label_map


# ─────────────────────────────────────────
# 0c. VISUALIZE BLOBS (SANITY CHECK)
# ─────────────────────────────────────────

def plot_blobs(cleaned, blobs, freqs,
               output_path="outputs/plots/blobs_detected.png"):
    """
    Save a plot of the cleaned spectrogram with bounding boxes
    drawn around each detected blob.

    Green box = detected signal region (ROI)
    This is your visual proof that segmentation is working.

    Args:
        cleaned     : 2D cleaned spectrogram (time_steps, freq_channels)
        blobs       : list of RegionProps from detect_blobs()
        freqs       : 1D frequency array in MHz
        output_path : where to save the PNG
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os

    print(f"[0c] Plotting blobs...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.imshow(
        cleaned,
        aspect="auto",
        origin="upper",
        cmap="inferno",
        extent=[freqs.min(), freqs.max(), cleaned.shape[0], 0]
    )

    # draw a bounding box around each detected blob
    freq_range  = freqs.max() - freqs.min()
    freq_per_px = freq_range / cleaned.shape[1]

    for blob in blobs:
        # regionprops gives pixel coords — convert to MHz for the x axis
        min_row, min_col, max_row, max_col = blob.bbox

        x      = freqs.min() + min_col * freq_per_px
        y      = min_row
        width  = (max_col - min_col) * freq_per_px
        height = max_row - min_row

        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)

        # label with SNR
        ax.text(
            x, y - 0.3,
            f"SNR {blob.mean_intensity:.0f}",
            color="lime",
            fontsize=7,
            va="bottom"
        )

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Time Step")
    ax.set_title(f"Blob Detection — {len(blobs)} signal(s) found")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Saved to: {output_path}")


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
    region_data  = data[:, 0, channel_indices]   # shape: (time_steps, n_channels)
    region_freqs = freqs[channel_indices]         # MHz values for those channels

    time_steps = region_data.shape[0]
    times      = np.arange(time_steps) * tsamp   # convert step index → seconds

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

    import sys
    sys.path.insert(0, "src")
    from preprocess import load_data, integrate_time, calculate_snr

    FILE = "data/raw/Voyager1.single_coarse.fine_res.h5"

    # Phase 1 pipeline
    wf, data, freqs  = load_data(FILE, f_start=8419.2, f_stop=8419.4)
    data_integrated  = integrate_time(data, n_steps=4)
    snr              = calculate_snr(data_integrated)

    # Phase 2 — blob detection
    cleaned          = denoise(data_integrated)
    blobs, label_map = detect_blobs(cleaned, threshold_sigma=3.0, min_area=5)
    plot_blobs(cleaned, blobs, freqs)

    # Phase 3 — feature extraction
    features, labels, regions = extract_features(
        data_integrated, freqs, snr,
        snr_threshold=10.0,
        tsamp=18.25 * 4
    )

    if features.size > 0:
        print("\n── Feature Matrix ──")
        print(f"{'':>3}  {'peak_freq':>12}  {'bandwidth':>10}  "
              f"{'snr':>8}  {'drift':>10}  {'R²':>6}")
        for i, row in enumerate(features):
            print(f"{i+1:>3}  {row[0]:>12.4f}  {row[1]:>10.1f}  "
                  f"{row[2]:>8.1f}  {row[3]:>10.4f}  {row[4]:>6.3f}")

    print("features.py ran successfully!")