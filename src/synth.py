import numpy as np

CLASSES = ['normal', 'tachy', 'brady', 'irregular']

def synth_window(kind: str, T: int = 60, base: int = 70, seed=None):
    rng = np.random.default_rng(seed)
    if kind == 'tachy':
        hr = rng.normal(130, 4, T)
    elif kind == 'brady':
        hr = rng.normal(40, 3, T)
    elif kind == 'irregular':
        hr = base + rng.normal(0, 5, T)
        for i in range(0, T, 7):
            hr[i:i+3] += rng.normal(0, 18, min(3, T - i))
    else:
        hr = rng.normal(base, 3, T)
    return np.clip(hr, 30, 200).astype('float32')

def make_dataset(n_per_class=200, T=60, seed=42):
    rng = np.random.default_rng(seed)
    X_seq, X_feat, y = [], [], []
    from .data import engineered_features_from_hr
    for ci, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            hr = synth_window(c, T=T, base=rng.integers(60, 80), seed=rng.integers(0, 1_000_000))
            z = (hr - hr.mean()) / (hr.std() + 1e-6)
            X_seq.append(z.astype('float32'))
            X_feat.append(engineered_features_from_hr(hr).astype('float32'))
            y.append(ci)
    return np.stack(X_seq), np.stack(X_feat), np.array(y, dtype='int64')