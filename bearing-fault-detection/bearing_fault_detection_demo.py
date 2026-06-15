"""
Bearing Fault Detection from Vibration Signals
===============================================
A machine-learning pipeline that classifies rolling-element bearing health from
vibration signals — the core of industrial predictive maintenance.

Four conditions are classified:
    - Normal (healthy)
    - Inner race fault
    - Outer race fault
    - Ball (rolling element) fault

A localised defect on a bearing surface produces a periodic series of mechanical
IMPACTS each time a rolling element strikes it. Each impact rings the bearing's
high-frequency structural resonance, producing a decaying oscillation. The REPETITION
RATE of those impacts is a known function of the bearing geometry and shaft speed —
the "characteristic defect frequency" — and it is DIFFERENT for inner-race, outer-race,
and ball faults. That is the physical signature this pipeline learns to recognise.

This demo generates physically-grounded synthetic signals (no data download required),
extracts diagnostic features (time-domain + envelope-spectrum), and trains a
Random Forest classifier. See the README for how to run it on the real CWRU dataset.

Dependencies: numpy, scipy, scikit-learn, matplotlib
Run: python bearing_fault_detection_demo.py

Author: Mohd Aaquib Khan
"""

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# BEARING GEOMETRY & CHARACTERISTIC DEFECT FREQUENCIES
# ─────────────────────────────────────────────────────────────────────────────
# Representative deep-groove ball bearing (similar to SKF 6205, used in CWRU data)
N_BALLS    = 9          # number of rolling elements
D_BALL     = 7.94       # ball diameter (mm)
D_PITCH    = 39.04      # pitch diameter (mm)
CONTACT_A  = 0.0        # contact angle (rad)

FS         = 12000      # sampling frequency (Hz) — matches CWRU
DURATION   = 1.0        # signal length (s)
N_SAMPLES  = int(FS * DURATION)
RESONANCE  = 3000.0     # bearing structural resonance excited by impacts (Hz)


def defect_frequencies(shaft_hz):
    """
    Characteristic bearing defect frequencies (Hz) for a given shaft speed.
    These standard formulas come from bearing kinematics.
    """
    ratio = (D_BALL / D_PITCH) * np.cos(CONTACT_A)
    bpfo = (N_BALLS / 2) * shaft_hz * (1 - ratio)            # outer race
    bpfi = (N_BALLS / 2) * shaft_hz * (1 + ratio)            # inner race
    bsf  = (D_PITCH / (2 * D_BALL)) * shaft_hz * (1 - ratio**2)   # ball spin
    ftf  = (shaft_hz / 2) * (1 - ratio)                      # cage / fundamental train
    return dict(BPFO=bpfo, BPFI=bpfi, BSF=bsf, FTF=ftf)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION (physics-grounded synthetic vibration)
# ─────────────────────────────────────────────────────────────────────────────
def impact_train(fault_hz, fs, n, decay=800.0, resonance=RESONANCE,
                 jitter=0.01, modulation=None, shaft_hz=None):
    """
    Build a series of decaying high-frequency impacts repeating at `fault_hz`.
    Each impact = exp(-decay*t) * sin(2*pi*resonance*t)  — a ringing structural response.

    modulation: None | 'shaft' | 'cage'
        Inner-race faults are amplitude-modulated at shaft speed (defect passes through
        the load zone once per revolution). Ball faults are modulated at cage frequency.
    """
    t = np.arange(n) / fs
    sig = np.zeros(n)
    period = 1.0 / fault_hz
    n_impacts = int(DURATION / period)

    for i in range(n_impacts):
        t0 = i * period * (1 + jitter * np.random.randn())   # slip/jitter
        idx = int(t0 * fs)
        if idx >= n:
            break
        tt = t[idx:] - t[idx]
        ring = np.exp(-decay * tt) * np.sin(2 * np.pi * resonance * tt)
        amp = 1.0
        if modulation == "shaft" and shaft_hz:
            amp = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * shaft_hz * t0))
        elif modulation == "cage" and shaft_hz:
            ftf = defect_frequencies(shaft_hz)["FTF"]
            amp = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * ftf * t0))
        sig[idx:] += amp * ring
    return sig


def generate_signal(condition, shaft_hz=None):
    """Generate one vibration signal for the given bearing condition."""
    if shaft_hz is None:
        shaft_hz = np.random.uniform(28, 32)        # ~1750 rpm +/- variation
    t = np.arange(N_SAMPLES) / FS
    f = defect_frequencies(shaft_hz)

    # Baseline: shaft harmonics (always present from imbalance/misalignment) + noise
    sig  = 0.20 * np.sin(2 * np.pi * shaft_hz * t)
    sig += 0.10 * np.sin(2 * np.pi * 2 * shaft_hz * t + 0.5)
    noise_level = np.random.uniform(0.08, 0.15)
    sig += noise_level * np.random.randn(N_SAMPLES)

    # Fault-specific impact train
    if condition == "Normal":
        pass
    elif condition == "Outer Race":
        # Stationary defect -> constant-amplitude impacts at BPFO
        sig += np.random.uniform(0.8, 1.2) * impact_train(f["BPFO"], FS, N_SAMPLES)
    elif condition == "Inner Race":
        # Defect rotates through load zone -> amplitude-modulated at shaft speed
        sig += np.random.uniform(0.8, 1.2) * impact_train(
            f["BPFI"], FS, N_SAMPLES, modulation="shaft", shaft_hz=shaft_hz)
    elif condition == "Ball":
        # Ball defect -> impacts at BSF, modulated at cage (FTF) frequency
        sig += np.random.uniform(0.7, 1.1) * impact_train(
            f["BSF"], FS, N_SAMPLES, modulation="cage", shaft_hz=shaft_hz, decay=600)
    return sig, shaft_hz


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def envelope_spectrum(sig, fs):
    """
    Envelope analysis — the standard bearing-diagnostic technique.
    The Hilbert transform extracts the signal envelope (the impact-rate modulation);
    its FFT reveals peaks at the characteristic defect frequencies.
    """
    analytic = hilbert(sig)
    env = np.abs(analytic)
    env = env - np.mean(env)
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(env), 1 / fs)
    return freqs, spec


def band_energy(freqs, spec, f_center, bw=8.0):
    """Energy of the envelope spectrum in a band around a target frequency."""
    if f_center <= 0:
        return 0.0
    mask = (freqs >= f_center - bw) & (freqs <= f_center + bw)
    return float(np.sum(spec[mask]**2))


def extract_features(sig, shaft_hz):
    """
    Build a feature vector mixing time-domain statistics (sensitive to impulsiveness)
    and envelope-spectrum energies at the characteristic defect frequencies.
    """
    feats = {}

    # ── Time-domain statistical features ──
    rms  = np.sqrt(np.mean(sig**2))
    peak = np.max(np.abs(sig))
    feats["rms"]           = rms
    feats["peak"]          = peak
    feats["std"]           = np.std(sig)
    feats["kurtosis"]      = kurtosis(sig)          # spikiness — key fault indicator
    feats["skewness"]      = skew(sig)
    feats["crest_factor"]  = peak / rms             # peak-to-RMS ratio
    feats["impulse_factor"]= peak / np.mean(np.abs(sig))
    feats["shape_factor"]  = rms / np.mean(np.abs(sig))

    # ── Envelope-spectrum features at characteristic defect frequencies ──
    freqs, spec = envelope_spectrum(sig, FS)
    f = defect_frequencies(shaft_hz)
    total = np.sum(spec**2) + 1e-12
    # normalised band energy at each fault frequency and its 2nd harmonic
    for name in ("BPFO", "BPFI", "BSF"):
        feats[f"env_{name}"]    = band_energy(freqs, spec, f[name])      / total
        feats[f"env_{name}_2x"] = band_energy(freqs, spec, 2 * f[name])  / total
    feats["env_shaft"] = band_energy(freqs, spec, shaft_hz) / total

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(n_per_class=150):
    conditions = ["Normal", "Outer Race", "Inner Race", "Ball"]
    X, y, raw_examples = [], [], {}
    print(f"Generating {n_per_class} samples x {len(conditions)} conditions ...")
    for cond in conditions:
        for i in range(n_per_class):
            sig, shaft_hz = generate_signal(cond)
            feats = extract_features(sig, shaft_hz)
            if i == 0:
                raw_examples[cond] = (sig, shaft_hz)     # keep one for plotting
            X.append(list(feats.values()))
            y.append(cond)
    feat_names = list(extract_features(*generate_signal("Normal")).keys())
    return np.array(X), np.array(y), feat_names, conditions, raw_examples


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
NAVY, BLUE = "#1F3A5F", "#2E609E"


def plot_signals_and_envelopes(raw_examples, conditions):
    fig, axes = plt.subplots(4, 2, figsize=(15, 11))
    fig.suptitle("Bearing Vibration Signals (left) and Envelope Spectra (right)",
                 fontsize=13, fontweight="bold", color=NAVY)
    colours = {"Normal": "#27AE60", "Outer Race": "#C0392B",
               "Inner Race": "#8E44AD", "Ball": "#E67E22"}

    for row, cond in enumerate(conditions):
        sig, shaft_hz = raw_examples[cond]
        t = np.arange(len(sig)) / FS
        f = defect_frequencies(shaft_hz)

        # Time signal (first 0.1 s)
        ax = axes[row, 0]
        n_show = int(0.1 * FS)
        ax.plot(t[:n_show], sig[:n_show], color=colours[cond], lw=0.7)
        ax.set_ylabel(cond, fontweight="bold", color=colours[cond])
        if row == 0:
            ax.set_title("Time-domain signal (first 0.1 s)", fontweight="bold")
        if row == 3:
            ax.set_xlabel("Time (s)")

        # Envelope spectrum
        ax = axes[row, 1]
        freqs, spec = envelope_spectrum(sig, FS)
        mask = freqs <= 400
        ax.plot(freqs[mask], spec[mask], color=colours[cond], lw=0.8)
        # mark characteristic frequencies
        for name, col in [("BPFO", "#C0392B"), ("BPFI", "#8E44AD"), ("BSF", "#E67E22")]:
            if f[name] <= 400:
                ax.axvline(f[name], color=col, ls="--", alpha=0.55, lw=1)
                ax.text(f[name], ax.get_ylim()[1]*0.82, name, rotation=90,
                        fontsize=7, color=col, va="top")
        if row == 0:
            ax.set_title("Envelope spectrum (defect freqs marked)", fontweight="bold")
        if row == 3:
            ax.set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig("signals_and_envelopes.png", dpi=140, bbox_inches="tight")
    print("  Saved: signals_and_envelopes.png")
    plt.close()


def plot_results(cm, conditions, clf, feat_names, acc):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Bearing Fault Classifier — Test Accuracy: {acc*100:.1f}%",
                 fontsize=13, fontweight="bold", color=NAVY)

    # Confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(conditions))); ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_yticks(range(len(conditions))); ax.set_yticklabels(conditions)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontweight="bold")
    thresh = cm.max() / 2
    for i in range(len(conditions)):
        for j in range(len(conditions)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Feature importance
    ax = axes[1]
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    ax.barh(range(len(imp)), imp[order][::-1], color=BLUE)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([feat_names[i] for i in order][::-1], fontsize=8)
    ax.set_title("Feature Importance (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig("classification_results.png", dpi=140, bbox_inches="tight")
    print("  Saved: classification_results.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("  BEARING FAULT DETECTION FROM VIBRATION SIGNALS")
    print("=" * 64)

    # Show example defect frequencies
    f_ex = defect_frequencies(30.0)
    print(f"\n  Defect frequencies at 30 Hz (1800 rpm) shaft speed:")
    for k, v in f_ex.items():
        print(f"    {k}: {v:6.2f} Hz")

    # Build dataset
    X, y, feat_names, conditions, raw_examples = build_dataset(n_per_class=150)
    print(f"\n  Dataset: {X.shape[0]} samples, {X.shape[1]} features each")

    # Train / test split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              stratify=y, random_state=42)

    # Train classifier
    print("\n  Training Random Forest classifier ...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    clf.fit(X_tr, y_tr)

    # Evaluate
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n  Test accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_te, y_pred, digits=3))

    cm = confusion_matrix(y_te, y_pred, labels=conditions)

    # Visualise
    print("  Rendering figures ...")
    plot_signals_and_envelopes(raw_examples, conditions)
    plot_results(cm, conditions, clf, feat_names, acc)

    print("\nDone — 2 figures generated.")
