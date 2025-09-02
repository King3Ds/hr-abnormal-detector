# src/data.py
import numpy as np
import pandas as pd


def read_any_heart_csv(fp) -> pd.DataFrame:
    """(Utility) Read either timestamp,hr or Fitbit(Time,Value) -> index by timestamp with 'hr' column."""
    df = pd.read_csv(fp)
    cols = {c.lower(): c for c in df.columns}
    if "timestamp" in cols and "hr" in cols:
        ts = pd.to_datetime(df[cols["timestamp"]], errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        hr = pd.to_numeric(df[cols["hr"]], errors="coerce")
        out = pd.DataFrame({"hr": hr.values}, index=ts).sort_index()
        return out
    if "time" in cols and "value" in cols:
        ts = pd.to_datetime(df[cols["time"]], errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        hr = pd.to_numeric(df[cols["value"]], errors="coerce")
        out = pd.DataFrame({"hr": hr.values}, index=ts).sort_index()
        return out
    raise ValueError("Unsupported CSV schema. Expected ['timestamp','hr'] or ['Time','Value'].")


def resample_1hz(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 1 Hz sampling & light interpolation."""
    df = df.copy()
    df = df[~df["hr"].isna()]
    out = df.resample("1s").mean()
    out["hr"] = out["hr"].interpolate(limit=5, limit_direction="both")
    return out


def hampel(series: pd.Series, k: int = 3, t0: float = 3.0) -> pd.Series:
    """Simple Hampel filter for outlier suppression."""
    x = series.copy()
    med = x.rolling(2 * k + 1, center=True).median()
    mad = (x - med).abs().rolling(2 * k + 1, center=True).median()
    thr = t0 * 1.4826 * mad
    mask = (x - med).abs() > thr
    x[mask] = med[mask]
    return x


def engineered_features_from_hr(hr_seq: np.ndarray) -> np.ndarray:
    """7 light-weight features used by the SVM."""
    hr = hr_seq.astype(float)
    slope = float((hr[-1] - hr[0]) / max(1, len(hr)))
    above120 = float(np.mean(hr > 120.0))
    below45 = float(np.mean(hr < 45.0))
    outlier_frac = float(np.mean(np.abs((hr - hr.mean()) / (hr.std() + 1e-6)) > 3))
    ibi = 60000.0 / np.clip(hr, 30, 220)  # ms
    diffs = np.diff(ibi)
    rmssd = float(np.sqrt(np.mean(diffs**2))) if len(diffs) > 0 else 0.0
    sdnn = float(np.std(ibi)) if len(ibi) > 1 else 0.0
    pnn50 = float(np.mean(np.abs(diffs) > 50.0)) if len(diffs) > 0 else 0.0
    return np.array([slope, above120, below45, outlier_frac, rmssd, sdnn, pnn50], dtype=np.float32)


def make_windows(hr: pd.Series, win: int = 60, stride: int = 30):
    """Legacy helper; not used by API (kept for training scripts)."""
    xs, feats, metas = [], [], []
    hr = hampel(hr.astype(float))
    n = len(hr)
    if n < win:
        return np.zeros((0, win), np.float32), np.zeros((0, 7), np.float32), []
    for start in range(0, n - win + 1, stride):
        seg = hr.iloc[start : start + win]
        if seg.isna().mean() > 0.2:
            continue
        z = (seg - seg.mean()) / (seg.std() + 1e-6)
        xs.append(z.values.astype("float32"))
        feats.append(engineered_features_from_hr(seg.values.astype("float32")))
        metas.append({"t0": seg.index[0], "t1": seg.index[-1]})
    X_seq = np.stack(xs) if xs else np.zeros((0, win), "float32")
    X_feat = np.stack(feats) if feats else np.zeros((0, 7), "float32")
    return X_seq, X_feat, metas


def to_windows(values, index, win_sec: int = 60, step_sec: int = 10):
    """
    Make overlapping windows from a 1 Hz HR series for inference.
    Returns:
      X_seq:  (n, win_sec)  z-scored sequences for CNN
      X_feat: (n, 7)        engineered features for SVM
      t_idx:  DatetimeIndex of window start times
    """
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) != len(index):
        raise ValueError("values and index must be the same length")

    win = int(win_sec)
    step = int(step_sec)
    if win <= 0 or step <= 0:
        raise ValueError("win_sec and step_sec must be positive")

    n = len(arr)
    if n < win:
        return np.empty((0, win), dtype=np.float32), np.empty((0, 7), dtype=np.float32), pd.DatetimeIndex([])

    X_seq, X_feat, t0 = [], [], []
    for start in range(0, n - win + 1, step):
        seg = arr[start : start + win]
        z = (seg - seg.mean()) / (seg.std() + 1e-6)
        X_seq.append(z.astype(np.float32))
        X_feat.append(engineered_features_from_hr(seg).astype(np.float32))
        t0.append(index[start])

    return np.stack(X_seq), np.stack(X_feat), pd.DatetimeIndex(t0)
