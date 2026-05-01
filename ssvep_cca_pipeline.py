#!/usr/bin/env python3
import csv
import os
import numpy as np

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from dataclasses import dataclass, replace
from typing import Optional, List, Tuple, Dict

from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, detrend
from sklearn.cross_decomposition import CCA


@dataclass
class Config:
    fs: float = 250.0

    # Windowing
    use_full_stim_5s: bool = False
    visual_latency_s: float = 0.14
    window_s: float = 2.0

    # CCA / FBCCA
    n_harmonics: int = 3
    n_subbands: int = 5        # FBCCA only

    # Preprocessing toggles
    do_detrend: bool = True
    do_notch: bool = False
    notch_hz: float = 50.0
    notch_q: float = 30.0
    do_bandpass: bool = True
    bp_low: float = 6.0
    bp_high: float = 45.0
    bp_order: int = 4
    do_car: bool = True

    # Channels (None = all)
    use_channels: Optional[List[int]] = None


#Data loading

def load_subject(mat_path: str) -> np.ndarray:
    d = loadmat(mat_path)
    if "data" not in d:
        raise KeyError(f"'data' not found in {mat_path}. Keys: {list(d.keys())}")
    data = np.asarray(d["data"], dtype=np.float64)
    if data.ndim != 4:
        raise ValueError(f"Expected 4-D 'data', got shape {data.shape}")
    return data


def load_freqs_from_freq_phase(dataset_dir: str) -> List[float]:
    fp_path = os.path.join(dataset_dir, "Freq_Phase.mat")
    if not os.path.exists(fp_path):
        raise FileNotFoundError(f"Missing {fp_path}.")
    d = loadmat(fp_path)
    candidate_keys = ["Freq", "freq", "freqs", "Freqs", "f", "F"]
    key = next((k for k in candidate_keys if k in d), None)
    if key is None:
        raise KeyError(f"Could not find frequency array in {fp_path}. Keys: {list(d.keys())}")
    freqs = np.squeeze(d[key]).astype(float)
    if freqs.size != 40:
        raise ValueError(f"Expected 40 frequencies, got {freqs.size}")
    return freqs.tolist()


def load_channel_labels(loc_path: str) -> List[str]:
    labels = []
    with open(loc_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                labels.append(parts[-1])
    return labels


def load_channel_indices(loc_path: str, wanted_labels: List[str]) -> List[int]:
    indices = []
    with open(loc_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[-1] in wanted_labels:
                indices.append(int(parts[0]) - 1)
    return indices


#Windowing & preprocessing

def extract_window(epoch: np.ndarray, cfg: Config) -> np.ndarray:
    """epoch: (n_channels, 1500) — Wang format: 0.5s pre + 5s stim + 0.5s post at 250 Hz."""
    if cfg.use_full_stim_5s:
        start = int(round(0.5 * cfg.fs))
        end = int(round(5.5 * cfg.fs))
        return epoch[:, start:end]
    stim_onset_s = 0.5
    start_s = stim_onset_s + cfg.visual_latency_s
    end_s = start_s + cfg.window_s
    start = int(round(start_s * cfg.fs))
    end = int(round(end_s * cfg.fs))
    if end > epoch.shape[-1]:
        raise ValueError("Window exceeds epoch length.")
    return epoch[:, start:end]


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


def notch_filter(x: np.ndarray, fs: float, f0: float, q: float) -> np.ndarray:
    b, a = iirnotch(w0=f0, Q=q, fs=fs)
    return filtfilt(b, a, x, axis=-1)


def car(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=0, keepdims=True)


def preprocess(x: np.ndarray, cfg: Config) -> np.ndarray:
    """x: (n_channels, n_samples)"""
    if cfg.do_detrend:
        x = detrend(x, axis=-1, type="linear")
    if cfg.do_notch:
        x = notch_filter(x, cfg.fs, cfg.notch_hz, cfg.notch_q)
    if cfg.do_bandpass:
        x = bandpass(x, cfg.fs, cfg.bp_low, cfg.bp_high, cfg.bp_order)
    if cfg.do_car:
        x = car(x)
    return x


#CCA

def make_ref(freq: float, fs: float, n_samples: int, n_harmonics: int) -> np.ndarray:
    t = np.arange(n_samples) / fs
    refs = []
    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))
    Y = np.stack(refs, axis=1)
    return Y - Y.mean(axis=0, keepdims=True)


def build_reference_bank(freqs: List[float], cfg: Config, n_samples: int) -> List[np.ndarray]:
    return [make_ref(f, cfg.fs, n_samples, cfg.n_harmonics) for f in freqs]


def cca_top_corr(X_trial: np.ndarray, Y_ref: np.ndarray, cca: CCA) -> float:
    """X_trial: (n_channels, n_samples), Y_ref: (n_samples, 2*n_harmonics)"""
    X = X_trial.T
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y_ref - Y_ref.mean(axis=0, keepdims=True)
    X_c, Y_c = cca.fit_transform(X, Y)
    r = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def cca_detect(X_trial: np.ndarray, ref_bank: List[np.ndarray], cca: CCA) -> int:
    scores = np.array([cca_top_corr(X_trial, Y, cca) for Y in ref_bank])
    return int(np.argmax(scores))


#FBCCA

def _fbcca_subbands(n: int, fs: float) -> List[Tuple[float, float]]:
    """Sub-bands [8n, 90] Hz for n=1..N (Chen et al. 2015)."""
    nyq = fs / 2.0 - 1.0
    return [(8.0 * k, min(90.0, nyq)) for k in range(1, n + 1)]


def fbcca_detect(X_trial: np.ndarray, ref_bank: List[np.ndarray], cfg: Config, cca: CCA) -> int:
    """
    Filter bank CCA (Chen et al. 2015).
    Bandpass-filters X into sub-bands, runs CCA in each, combines with
    weights w_k = k^(-1.25) + 0.25, classifies by argmax of weighted sum.
    """
    subbands = _fbcca_subbands(cfg.n_subbands, cfg.fs)
    scores = np.zeros(len(ref_bank))

    for k, (low, high) in enumerate(subbands):
        w = (k + 1) ** (-1.25) + 0.25
        X_band = bandpass(X_trial, cfg.fs, low, high, order=4)
        for i, Y in enumerate(ref_bank):
            r = cca_top_corr(X_band, Y, cca)
            scores[i] += w * (r ** 2)

    return int(np.argmax(scores))


#PSDA

def _snr_at_freq(psd: np.ndarray, freq_bins: np.ndarray, target_hz: float, n_neighbors: int = 2) -> float:
    idx = int(np.argmin(np.abs(freq_bins - target_hz)))
    signal = psd[idx]
    neighbor_idx = [
        idx + d for d in range(-n_neighbors, n_neighbors + 1)
        if d != 0 and 0 <= idx + d < len(psd)
    ]
    noise = np.mean(psd[neighbor_idx]) if neighbor_idx else 1.0
    return float(signal / noise) if noise > 0 else 0.0


def psda_detect(X_trial: np.ndarray, freqs: List[float], cfg: Config) -> int:
    """
    Power Spectral Density Analysis. Averages PSD across channels,
    scores each target by mean SNR across harmonics, classifies by argmax.
    """
    n = X_trial.shape[1]
    freq_bins = np.fft.rfftfreq(n, d=1.0 / cfg.fs)
    psd = np.mean(np.abs(np.fft.rfft(X_trial, axis=1)) ** 2, axis=0)

    scores = np.zeros(len(freqs))
    for i, freq in enumerate(freqs):
        snrs = [_snr_at_freq(psd, freq_bins, freq * h) for h in range(1, cfg.n_harmonics + 1)]
        scores[i] = float(np.mean(snrs))

    return int(np.argmax(scores))


#Subject-level evaluation

def _eval_subject(data: np.ndarray, cfg: Config, freqs: List[float], method: str) -> float:
    n_ch, _, n_targets, n_blocks = data.shape
    ch_idx = cfg.use_channels if cfg.use_channels is not None else list(range(n_ch))
    n_samples = int(round((5.0 if cfg.use_full_stim_5s else cfg.window_s) * cfg.fs))

    ref_bank = build_reference_bank(freqs, cfg, n_samples)
    cca = CCA(n_components=1, max_iter=2000)

    correct = 0
    total = 0
    for target in range(n_targets):
        for block in range(n_blocks):
            epoch = data[:, :, target, block][ch_idx, :]
            win = extract_window(epoch, cfg)
            win = preprocess(win, cfg)

            if method == "cca":
                pred = cca_detect(win, ref_bank, cca)
            elif method == "fbcca":
                pred = fbcca_detect(win, ref_bank, cfg, cca)
            elif method == "psda":
                pred = psda_detect(win, freqs, cfg)
            else:
                raise ValueError(f"Unknown method: {method}")

            correct += int(pred == target)
            total += 1

    return correct / total


#SNR electrode map

def _subject_electrode_snr(data: np.ndarray, cfg: Config, freqs: List[float]) -> np.ndarray:
    """
    Returns (n_channels,): mean SNR per electrode across all trials at the
    correct target frequency. Uses FFT, not CCA — fast and appropriate for
    single-channel comparison.
    """
    n_ch, _, n_targets, n_blocks = data.shape
    snr_sum = np.zeros(n_ch)
    count = 0

    for target in range(n_targets):
        target_freq = freqs[target]
        for block in range(n_blocks):
            epoch = data[:, :, target, block]
            win = extract_window(epoch, cfg)
            win = preprocess(win, cfg)
            n = win.shape[1]
            freq_bins = np.fft.rfftfreq(n, d=1.0 / cfg.fs)

            for ch in range(n_ch):
                psd = np.abs(np.fft.rfft(win[ch])) ** 2
                snrs = [_snr_at_freq(psd, freq_bins, target_freq * h) for h in range(1, cfg.n_harmonics + 1)]
                snr_sum[ch] += float(np.mean(snrs))
            count += 1

    return snr_sum / count


def electrode_snr_all_subjects(
    dataset_dir: str,
    cfg: Config,
    freqs: List[float],
    channel_labels: List[str],
    n_jobs: int = -1,
) -> Dict[str, List[float]]:
    subject_files = sorted(f for f in os.listdir(dataset_dir) if f.startswith("S") and f.endswith(".mat"))
    mat_paths = [os.path.join(dataset_dir, f) for f in subject_files]

    def _worker(path):
        data = load_subject(path)
        snr = _subject_electrode_snr(data, cfg, freqs)
        print(f"SNR map done: {os.path.basename(path)}")
        return snr

    results = Parallel(n_jobs=n_jobs)(delayed(_worker)(p) for p in mat_paths)

    per_channel: Dict[str, List[float]] = {label: [] for label in channel_labels}
    for snr_arr in results:
        for ch_idx, label in enumerate(channel_labels):
            per_channel[label].append(float(snr_arr[ch_idx]))

    return per_channel


#Method comparison

def compare_methods(
    dataset_dir: str,
    cfg: Config,
    freqs: List[float],
    channel_indices: List[int],
    methods: List[str],
    n_jobs: int = -1,
) -> Dict[str, List[float]]:
    subject_files = sorted(f for f in os.listdir(dataset_dir) if f.startswith("S") and f.endswith(".mat"))
    mat_paths = [os.path.join(dataset_dir, f) for f in subject_files]
    cfg_m = replace(cfg, use_channels=channel_indices)

    def _worker(path):
        data = load_subject(path)
        accs = {m: _eval_subject(data, cfg_m, freqs, method=m) for m in methods}
        print(f"Methods done: {os.path.basename(path)}")
        return accs

    results = Parallel(n_jobs=n_jobs)(delayed(_worker)(p) for p in mat_paths)

    per_method: Dict[str, List[float]] = {m: [] for m in methods}
    for r in results:
        for m in methods:
            per_method[m].append(r[m])

    return per_method


#Plotting & output 

def summarize_region(per_channel: Dict[str, List[float]], labels: List[str]) -> Tuple[float, float]:
    vals = [v for l in labels for v in per_channel.get(l, [])]
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals)
    return float(np.mean(arr)), float(np.std(arr))


def plot_snr_channel_means(per_channel: Dict[str, List[float]], save_path: str):
    labels = list(per_channel.keys())
    means = [np.mean(per_channel[l]) for l in labels]
    plt.figure(figsize=(16, 5))
    plt.bar(labels, means)
    plt.xticks(rotation=90)
    plt.ylabel("Mean SNR")
    plt.title("Single-electrode SNR by channel")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_snr_region_boxplot(per_channel: Dict[str, List[float]], save_path: str):
    groups = {
        "Occipital": ["O1", "Oz", "O2"],
        "Parietal": ["Pz", "P3", "P4"],
        "Temporal": ["T7", "T8", "TP7", "TP8", "P7", "P8"],
    }
    data, names = [], []
    for name, labels in groups.items():
        vals = [v for l in labels for v in per_channel.get(l, [])]
        if vals:
            data.append(vals)
            names.append(name)
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=names)
    plt.ylabel("SNR")
    plt.title("Single-electrode SNR by brain region")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_method_comparison(per_method: Dict[str, List[float]], save_path: str):
    methods = list(per_method.keys())
    means = [np.mean(per_method[m]) for m in methods]
    stds = [np.std(per_method[m]) for m in methods]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(methods, means, yerr=stds, capsize=5)
    for bar, m in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{m:.3f}", ha="center")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.title("Method comparison — posterior montage, 2 s window")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_snr_summary_csv(per_channel: Dict[str, List[float]], save_path: str):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "mean_snr", "std_snr", "n_subjects"])
        for label, vals in per_channel.items():
            writer.writerow([label, float(np.mean(vals)), float(np.std(vals)), len(vals)])


def save_method_summary_csv(per_method: Dict[str, List[float]], save_path: str):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "mean_accuracy", "std_accuracy", "n_subjects"])
        for method, vals in per_method.items():
            writer.writerow([method, float(np.mean(vals)), float(np.std(vals)), len(vals)])


#Main

def main():
    dataset_dir = os.environ.get("WANG_DATASET_DIR", "").strip()
    if not dataset_dir:
        raise SystemExit("Set WANG_DATASET_DIR to folder containing S1.mat ... S35.mat and Freq_Phase.mat")

    output_dir = os.environ.get("SSVEP_OUTPUT_DIR", ".").strip()
    os.makedirs(output_dir, exist_ok=True)

    freqs = load_freqs_from_freq_phase(dataset_dir)
    print("First 10 target freqs:", freqs[:10])

    loc_path = os.path.join(dataset_dir, "64-channels.loc")
    channel_labels = load_channel_labels(loc_path)

    posterior_labels = ["Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"]
    posterior_indices = load_channel_indices(loc_path, posterior_labels)

    cfg = Config(
        use_full_stim_5s=False,
        window_s=2.0,
        n_harmonics=3,
        n_subbands=5,
        do_detrend=True,
        do_bandpass=True,
        bp_low=6.0,
        bp_high=45.0,
        do_car=True,
    )

    #SNR-based electrode comparison
    print("\n=== Electrode SNR map (all 64 channels) ===")
    cfg_snr = replace(cfg, do_car=False)   # CAR would remove signal common across channels
    per_channel_snr = electrode_snr_all_subjects(
        dataset_dir=dataset_dir,
        cfg=cfg_snr,
        freqs=freqs,
        channel_labels=channel_labels,
    )
    save_snr_summary_csv(per_channel_snr, os.path.join(output_dir, "electrode_snr_summary.csv"))
    plot_snr_channel_means(per_channel_snr, os.path.join(output_dir, "electrode_snr_means.png"))
    plot_snr_region_boxplot(per_channel_snr, os.path.join(output_dir, "electrode_snr_boxplot.png"))

    occipital_labels = ["O1", "Oz", "O2"]
    parietal_labels  = ["Pz", "P3", "P4"]
    temporal_labels  = ["T7", "T8", "TP7", "TP8", "P7", "P8"]

    occ_m, occ_s = summarize_region(per_channel_snr, occipital_labels)
    par_m, par_s = summarize_region(per_channel_snr, parietal_labels)
    tmp_m, tmp_s = summarize_region(per_channel_snr, temporal_labels)
    print(f"Occipital SNR : mean={occ_m:.3f}, std={occ_s:.3f}")
    print(f"Parietal  SNR : mean={par_m:.3f}, std={par_s:.3f}")
    print(f"Temporal  SNR : mean={tmp_m:.3f}, std={tmp_s:.3f}")

    #CCA vs FBCCA vs PSDA on posterior montage
    print("\n=== Method comparison (posterior montage) ===")
    per_method = compare_methods(
        dataset_dir=dataset_dir,
        cfg=cfg,
        freqs=freqs,
        channel_indices=posterior_indices,
        methods=["cca", "fbcca", "psda"],
    )
    save_method_summary_csv(per_method, os.path.join(output_dir, "method_comparison.csv"))
    plot_method_comparison(per_method, os.path.join(output_dir, "method_comparison.png"))

    for method, vals in per_method.items():
        print(f"{method.upper():6s} : mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

    print(f"\nOutputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
