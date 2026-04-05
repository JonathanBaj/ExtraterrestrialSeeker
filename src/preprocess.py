# src/preprocess.py

import numpy as np
from blimpy import Waterfall

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

def load_data(filepath, f_start=None, f_stop=None):
    """
    Load an .h5 or .fil file using blimpy.
    Optionally slice a specific frequency range (in MHz).

    Args:
        filepath (str): Path to your .h5 or .fil file
        f_start  (float): Start frequency in MHz (optional)
        f_stop   (float): Stop  frequency in MHz (optional)

    Returns:
        wf   : blimpy Waterfall object (holds header + raw data)
        data : numpy array of shape (time_steps, 1, frequency_channels)
        freqs: numpy array of frequency values in MHz
    """
    print(f"[1/3] Loading file: {filepath}")

    # Waterfall is blimpy's main class for reading .h5 and .fil files
    # f_start / f_stop let you grab just a slice instead of the whole file
    wf = Waterfall(filepath, f_start=f_start, f_stop=f_stop)

    wf.info()                          # prints header info to terminal

    data  = wf.data                    # shape: (time_steps, 1, freq_channels)
    freqs = wf.get_freqs()             # 1D array of frequency values in MHz

    print(f"    Data shape : {data.shape}")
    print(f"    Freq range (slice): {wf.container.f_start:.4f} – {wf.container.f_stop:.4f} MHz")

    return wf, data, freqs


# ─────────────────────────────────────────
# 2. TIME INTEGRATION
# ─────────────────────────────────────────

def integrate_time(data, n_steps=4):
    """
    Average every n_steps time rows together.
    This reduces file size and smooths out short noise spikes.

    Example: 16 time steps + n_steps=4 → 4 averaged rows

    Args:
        data    : numpy array shape (time_steps, 1, freq_channels)
        n_steps : how many time rows to average together

    Returns:
        integrated: numpy array shape (time_steps // n_steps, 1, freq_channels)
    """
    print(f"[2/3] Integrating time steps (averaging every {n_steps} rows)...")

    time_steps = data.shape[0]

    # Trim so time_steps divides evenly by n_steps
    trim       = time_steps - (time_steps % n_steps)
    data       = data[:trim]

    # Reshape into blocks, then average each block
    integrated = data.reshape(-1, n_steps, data.shape[1], data.shape[2]).mean(axis=1)

    print(f"    Before: {time_steps} time steps")
    print(f"    After : {integrated.shape[0]} time steps")

    return integrated


# ─────────────────────────────────────────
# 3. SNR CALCULATION
# ─────────────────────────────────────────

def calculate_snr(data):
    """
    Compute the Signal-to-Noise Ratio for every frequency channel.

    Method:
        - Signal   = mean power at each frequency channel across time
        - Noise    = standard deviation across time (how much it fluctuates)
        - SNR      = Signal / Noise  (per channel)

    A high SNR means something real and structured is there.
    A low  SNR means it's just random noise / static.

    Args:
        data : numpy array shape (time_steps, 1, freq_channels)

    Returns:
        snr  : 1D numpy array of SNR values, one per frequency channel
    """
    print("[3/3] Calculating SNR per frequency channel...")

    # Squeeze out the middle dimension → shape (time_steps, freq_channels)
    d = data[:, 0, :]

    signal = d.mean(axis=0)    # average power per channel across time
    noise  = d.std(axis=0)     # how much each channel fluctuates across time

    # Avoid dividing by zero in dead/empty channels
    noise  = np.where(noise == 0, 1e-10, noise)

    snr    = signal / noise

    print(f"    Max SNR : {snr.max():.2f}")
    print(f"    Mean SNR: {snr.mean():.2f}")
    print(f"    Channels above SNR 10: {(snr > 10).sum()}")

    return snr


# ─────────────────────────────────────────
# 4. WATERFALL PLOT (SANITY CHECK)
# ─────────────────────────────────────────

def plot_waterfall(filepath, f_start=8419.0, f_stop=8421.0,
                   output_path="outputs/plots/voyager_test.png"):
    """
    Generate and save a waterfall (spectrogram) plot.

    What you're looking for:
        - Diagonal line = Voyager! (real signal with Doppler drift)
        - Horizontal bars = RFI from satellites/ground transmitters
        - Vertical blocks = broadband interference (nearby electronics)

    Args:
        filepath    : path to your .h5 file
        f_start     : start frequency in MHz
        f_stop      : stop  frequency in MHz
        output_path : where to save the PNG
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import os

    print(f"[4/4] Generating waterfall plot...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    wf = Waterfall(filepath, f_start=f_start, f_stop=f_stop)
    data  = wf.data[:, 0, :]
    freqs = wf.get_freqs()

    # Log-normalize to reveal faint structure
    vmin = np.percentile(data, 5)
    vmax = np.percentile(data, 99.9)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        extent=[freqs.min(), freqs.max(), data.shape[0], 0],
        norm=mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
        cmap="inferno"
    )
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Time Step")
    ax.set_title("Voyager 1 — Waterfall Plot (log scale)")
    plt.colorbar(ax.images[0], ax=ax, label="Power (log)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Saved to: {output_path}")


# ─────────────────────────────────────────
# QUICK TEST  (run this file directly to test it)
# ─────────────────────────────────────────

if __name__ == "__main__":

    FILE = "data/raw/Voyager1.single_coarse.fine_res.h5"

    # Step 1 — load (grab a small frequency slice around Voyager's known signal)
    wf, data, freqs = load_data(FILE, f_start=8419.2, f_stop=8419.4)

    # Step 2 — integrate time
    data_integrated = integrate_time(data, n_steps=4)

    # Step 3 — SNR
    snr = calculate_snr(data_integrated)

    # Step 4 — waterfall plot
    plot_waterfall(FILE, f_start=8419.2, f_stop=8419.4)

    print("\n✅ preprocess.py ran successfully!")