#!/usr/bin/env python3
#!/usr/bin/env python3
import csv
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # important for headless cluster jobs
import matplotlib.pyplot as plt

from dataclasses import dataclass, replace
from typing import Optional, List, Tuple

from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, detrend
from sklearn.cross_decomposition import CCA


@dataclass
class Config:
    fs: float = 250.0

    # Windowing
    use_full_stim_5s: bool = True
    visual_latency_s: float = 0.14   # used only if use_full_stim_5s=False
    window_s: float = 2.0            # used only if use_full_stim_5s=False

    # CCA
    n_harmonics: int = 5

    # Preprocessing toggles
    do_detrend: bool = False
    do_notch: bool = False
    notch_hz: float = 50.0
    notch_q: float = 30.0
    do_bandpass: bool = False
    bp_low: float = 6.0
    bp_high: float = 45.0
    bp_order: int = 4
    do_car: bool = False

    # Channels (None = all)
    use_channels: Optional[List[int]] = None


def load_subject(mat_path: str) -> np.ndarray:
    d = loadmat(mat_path)
    if "data" not in d:
        raise KeyError(f"'data' not found in {mat_path}. Keys: {list(d.keys())}")
    data = np.asarray(d["data"], dtype=np.float64)
    if data.ndim != 4:
        raise ValueError(f"Expected 4-D 'data', got shape {data.shape}")
    return data


def load_freqs_from_freq_phase(dataset_dir: str) -> List[float]:
    """
    Loads per-target frequencies from Freq_Phase.mat so that
    freqs[target_index] matches the dataset target ordering.
    """
    fp_path = os.path.join(dataset_dir, "Freq_Phase.mat")
    if not os.path.exists(fp_path):
        raise FileNotFoundError(
            f"Missing {fp_path}. Download 'Freq_Phase.mat' and put it in your dataset directory."
        )

    d = loadmat(fp_path)

    # Inspect keys if this fails; common keys are often 'freqs' or similar.
    candidate_keys = ["Freq", "freq", "freqs", "Freqs", "f", "F"]
    key = next((k for k in candidate_keys if k in d), None)

    if key is None:
        raise KeyError(f"Could not find frequency array in {fp_path}. Keys: {list(d.keys())}")

    freqs = np.squeeze(d[key]).astype(float)

    if freqs.size != 40:
        raise ValueError(f"Expected 40 frequencies, got {freqs.size} from key '{key}'")

    return freqs.tolist()


def extract_window(epoch: np.ndarray, cfg: Config) -> np.ndarray:
    """
    epoch: (n_channels, 1500)
    Wang epochs: 0.5s pre, 5s stim, 0.5s post at 250Hz => 1500 samples.
    """
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

def notch(x: np.ndarray, fs: float, f0: float, q: float) -> np.ndarray:
    b, a = iirnotch(w0=f0, Q=q, fs=fs)
    return filtfilt(b, a, x, axis=-1)

def car(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=0, keepdims=True)

def preprocess(x: np.ndarray, cfg: Config) -> np.ndarray:
    """
    x: (n_channels, n_samples_window)
    """
    if cfg.do_detrend:
        x = detrend(x, axis=-1, type="linear")
    if cfg.do_notch:
        x = notch(x, cfg.fs, cfg.notch_hz, cfg.notch_q)
    if cfg.do_bandpass:
        x = bandpass(x, cfg.fs, cfg.bp_low, cfg.bp_high, cfg.bp_order)
    if cfg.do_car:
        x = car(x)
    return x


# ---------- CCA ----------
def make_ref(freq: float, fs: float, n_samples: int, n_harmonics: int) -> np.ndarray:
    t = np.arange(n_samples) / fs
    refs = []
    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))
    Y = np.stack(refs, axis=1)
    Y = Y - Y.mean(axis=0, keepdims=True)
    return Y

def build_reference_bank(freqs: List[float], cfg: Config, n_samples: int) -> List[np.ndarray]:
    return [make_ref(f, cfg.fs, n_samples, cfg.n_harmonics) for f in freqs]

def cca_top_corr(X_trial: np.ndarray, Y_ref: np.ndarray, cca: CCA) -> float:
    """
    X_trial: (n_channels, n_samples)
    Y_ref:   (n_samples, n_ref_features)
    """
    X = X_trial.T
    Y = Y_ref

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    X_c, Y_c = cca.fit_transform(X, Y)

    r = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    if np.isnan(r):
        return 0.0
    return float(r)

def detect(
    X_trial: np.ndarray,
    freqs: List[float],
    ref_bank: List[np.ndarray],
    cca: CCA,
) -> Tuple[int, float, np.ndarray]:
    scores = np.zeros(len(freqs), dtype=float)
    for i, Y in enumerate(ref_bank):
        scores[i] = cca_top_corr(X_trial, Y, cca)
    best_idx = int(np.argmax(scores))
    return best_idx, freqs[best_idx], scores

def evaluate_subject(data: np.ndarray, cfg: Config, freqs: List[float]) -> float:
    n_ch, n_t, n_targets, n_blocks = data.shape
    ch_idx = cfg.use_channels if cfg.use_channels is not None else list(range(n_ch))

    n_samples = int(round(5.0 * cfg.fs)) if cfg.use_full_stim_5s else int(round(cfg.window_s * cfg.fs))
    ref_bank = build_reference_bank(freqs, cfg, n_samples)
    cca = CCA(n_components=1, max_iter=2000)

    correct = 0
    total = 0

    for target in range(n_targets):
        for block in range(n_blocks):
            epoch = data[:, :, target, block]      # (64, 1500)
            epoch = epoch[ch_idx, :]               # channel subset (or all)

            win = extract_window(epoch, cfg)
            win = preprocess(win, cfg)

            pred_idx, pred_f, scores = detect(win, freqs, ref_bank, cca)

            correct += int(pred_idx == target)
            total += 1

    return correct / total

def evaluate_subject_with_channels(
    data: np.ndarray,
    cfg: Config,
    freqs: List[float],
    channel_indices: List[int],
) -> float:
    cfg_local = replace(cfg, use_channels=channel_indices)
    cfg_proc = replace(cfg_local, do_car=False)

    n_samples = int(round(5.0 * cfg.fs)) if cfg.use_full_stim_5s else int(round(cfg.window_s * cfg.fs))
    ref_bank = build_reference_bank(freqs, cfg_proc, n_samples)
    cca = CCA(n_components=1, max_iter=2000)

    *_, n_targets, n_blocks = data.shape
    correct = 0
    total = 0

    for target in range(n_targets):
        for block in range(n_blocks):
            epoch = data[:, :, target, block]

            # CAR applied before subsetting so it uses all channels
            if cfg_local.do_car:
                epoch = car(epoch)

            epoch = epoch[channel_indices, :]
            win = extract_window(epoch, cfg_local)
            win = preprocess(win, cfg_proc)

            pred_idx, _, _ = detect(win, freqs, ref_bank, cca)

            correct += int(pred_idx == target)
            total += 1

    return correct / total


def evaluate_single_electrodes(
    data: np.ndarray,
    cfg: Config,
    freqs: List[float],
    channel_labels: List[str],
) -> dict[str, float]:
    results = {}

    # For single-electrode analysis, disable CAR unless applied before subsetting
    cfg_single = replace(cfg, do_car=False)

    for ch_idx, ch_label in enumerate(channel_labels):
        acc = evaluate_subject_with_channels(
            data=data,
            cfg=cfg_single,
            freqs=freqs,
            channel_indices=[ch_idx],
        )
        results[ch_label] = acc

    return results


def _process_subject(mat_path: str, cfg: Config, freqs: List[float], channel_labels: List[str]) -> dict[str, float]:
    data = load_subject(mat_path)
    result = evaluate_single_electrodes(data=data, cfg=cfg, freqs=freqs, channel_labels=channel_labels)
    print(f"Finished single-electrode analysis for {os.path.basename(mat_path)}")
    return result


def evaluate_all_subjects_single_electrodes(
    dataset_dir: str,
    cfg: Config,
    freqs: List[float],
    channel_labels: List[str],
    n_jobs: int = -1,
) -> dict[str, list[float]]:
    subject_files = sorted(
        f for f in os.listdir(dataset_dir)
        if f.startswith("S") and f.endswith(".mat")
    )
    mat_paths = [os.path.join(dataset_dir, f) for f in subject_files]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_subject)(p, cfg, freqs, channel_labels) for p in mat_paths
    )

    per_channel: dict[str, list[float]] = {label: [] for label in channel_labels}
    for single_results in results:
        for label, acc in single_results.items():
            per_channel[label].append(acc)

    return per_channel


def load_channel_indices(loc_path: str, wanted_labels: list[str]) -> list[int]:
    indices = []
    with open(loc_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            ch_num = int(parts[0]) - 1  
            label = parts[-1]
            if label in wanted_labels:
                indices.append(ch_num)
    return indices

def load_channel_labels(loc_path: str) -> list[str]:
    labels = []
    with open(loc_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            label = parts[-1]
            labels.append(label)
    return labels


def summarize_region(per_channel: dict[str, list[float]], labels: list[str]) -> tuple[float, float]:
    vals = []
    for label in labels:
        if label in per_channel:
            vals.extend(per_channel[label])
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def plot_channel_means(per_channel: dict[str, list[float]], save_path: str):
    labels = list(per_channel.keys())
    means = [np.mean(per_channel[l]) for l in labels]

    plt.figure(figsize=(16, 5))
    plt.bar(labels, means)
    plt.xticks(rotation=90)
    plt.ylabel("Mean accuracy")
    plt.title("Single-electrode CCA accuracy by channel")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_region_boxplot(per_channel: dict[str, list[float]], save_path: str):
    groups = {
        "Occipital": ["O1", "Oz", "O2"],
        "Parietal": ["Pz", "P3", "P4"],
        "Temporal": ["T7", "T8", "TP7", "TP8", "P7", "P8"],
    }

    data = []
    names = []
    for name, labels in groups.items():
        vals = []
        for label in labels:
            if label in per_channel:
                vals.extend(per_channel[label])
        if vals:
            data.append(vals)
            names.append(name)

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=names)
    plt.ylabel("Accuracy")
    plt.title("CCA accuracy by electrode region")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_channel_summary_csv(per_channel: dict[str, list[float]], save_path: str):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "mean_accuracy", "std_accuracy", "n_subjects"])
        for label, vals in per_channel.items():
            writer.writerow([
                label,
                float(np.mean(vals)),
                float(np.std(vals)),
                len(vals),
            ])


def main():
    dataset_dir = os.environ.get("WANG_DATASET_DIR", "").strip()
    if not dataset_dir:
        raise SystemExit("Set WANG_DATASET_DIR to folder containing S1.mat ... S35.mat and Freq_Phase.mat")

    output_dir = os.environ.get("SSVEP_OUTPUT_DIR", ".").strip()
    os.makedirs(output_dir, exist_ok=True)

    freqs = load_freqs_from_freq_phase(dataset_dir)
    print("First 10 target freqs from Freq_Phase.mat:", freqs[:10])

    loc_path = os.path.join(dataset_dir, "64-channels.loc")
    channel_labels = load_channel_labels(loc_path)

    posterior_labels = ["Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"]
    posterior_indices = load_channel_indices(loc_path, posterior_labels)

    cfg = Config(
        use_full_stim_5s=False,
        window_s=2.0,
        n_harmonics=3,
        do_detrend=True,
        do_bandpass=True,
        bp_low=6.0,
        bp_high=45.0,
        do_car=True,
        use_channels=posterior_indices
    )

    subject_files = sorted(
        f for f in os.listdir(dataset_dir)
        if f.startswith("S") and f.endswith(".mat")
    )

    # 1) Baseline posterior montage
    all_acc = []
    for subject_file in subject_files:
        mat_path = os.path.join(dataset_dir, subject_file)
        data = load_subject(mat_path)
        acc = evaluate_subject_with_channels(data, cfg, freqs, posterior_indices)
        all_acc.append(acc)
        print(f"[posterior montage] {subject_file}: {acc:.3f}")

    print("\n=== Posterior Montage Results ===")
    print(f"Mean accuracy: {np.mean(all_acc):.3f}")
    print(f"Std accuracy:  {np.std(all_acc):.3f}")

    # 2) Single-electrode analysis
    per_channel = evaluate_all_subjects_single_electrodes(
        dataset_dir=dataset_dir,
        cfg=cfg,
        freqs=freqs,
        channel_labels=channel_labels
    )

    save_channel_summary_csv(
        per_channel,
        os.path.join(output_dir, "single_electrode_summary.csv")
    )
    plot_channel_means(
        per_channel,
        os.path.join(output_dir, "single_electrode_means.png")
    )
    plot_region_boxplot(
        per_channel,
        os.path.join(output_dir, "region_boxplot.png")
    )

    # 3) Region summaries
    occipital_labels = ["O1", "Oz", "O2"]
    parietal_labels = ["Pz", "P3", "P4"]
    temporal_labels = ["T7", "T8", "TP7", "TP8", "P7", "P8"]

    occ_mean, occ_std = summarize_region(per_channel, occipital_labels)
    par_mean, par_std = summarize_region(per_channel, parietal_labels)
    tmp_mean, tmp_std = summarize_region(per_channel, temporal_labels)

    print("\n=== Region Summary ===")
    print(f"Occipital ({occipital_labels}): mean={occ_mean:.3f}, std={occ_std:.3f}")
    print(f"Parietal  ({parietal_labels}): mean={par_mean:.3f}, std={par_std:.3f}")
    print(f"Temporal  ({temporal_labels}): mean={tmp_mean:.3f}, std={tmp_std:.3f}")
    print("\nSaved:")
    print(f"- {os.path.join(output_dir, 'single_electrode_summary.csv')}")
    print(f"- {os.path.join(output_dir, 'single_electrode_means.png')}")
    print(f"- {os.path.join(output_dir, 'region_boxplot.png')}")


if __name__ == "__main__":
    main()