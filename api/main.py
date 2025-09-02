# api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import io, os, zipfile, gzip, tempfile, re, warnings

import numpy as np
import pandas as pd
import joblib
import torch

from src.models.cnn import HR1DCNN
from src.data import to_windows
from src.synth import CLASSES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
WEB = ROOT / "web"

app = FastAPI(title="HR Abnormality Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Frontend --------------------
@app.get("/")
def index():
    idx = WEB / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="index.html missing in /web")
    return FileResponse(idx)

# -------------------- Models ----------------------
svm = joblib.load(MODELS / "svm.joblib")
meta = joblib.load(MODELS / "fuser.joblib")

cnn = HR1DCNN(n_classes=len(CLASSES))
state = torch.load(MODELS / "cnn.pt", map_location="cpu")
try:
    cnn.load_state_dict(state)
except Exception:
    # backward-compat: rename old layer keys
    rename = {}
    for k in state.keys():
        nk = k
        if k.startswith("fe."): nk = k.replace("fe.", "net.")
        if k.startswith("cls.4"): nk = k.replace("cls.4", "fc")
        if nk != k: rename[nk] = state[k]
    if rename:
        cnn.load_state_dict(rename, strict=False)
cnn.eval()

# -------------------- Upload helpers --------------
MAX_UPLOAD_MB = 200
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

def _save_stream_to_tmp(up: UploadFile) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    total = 0
    try:
        while True:
            chunk = up.file.read(1024 * 1024)  # 1 MB
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                tmp.close(); os.unlink(tmp.name)
                raise HTTPException(413, f"Upload too large (> {MAX_UPLOAD_MB} MB).")
            tmp.write(chunk)
        tmp.flush()
        return tmp.name
    finally:
        up.file.close()

def _open_any_csv(path: str, filename: str):
    name = (filename or "").lower()
    if name.endswith(".gz"):
        return gzip.open(path, "rb")
    if name.endswith(".zip"):
        zf = zipfile.ZipFile(path, "r")
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise HTTPException(400, "ZIP has no CSV inside.")
        return io.BytesIO(zf.read(csvs[0]))
    return open(path, "rb")

# -------------------- Time parsing ----------------
# We avoid generic parse (which emits the warning) by trying explicit formats in order.
_FITBIT_FORMATS = [
    "%m/%d/%Y %I:%M:%S %p",       # 04/12/2016 11:59:59 PM
    "%m/%d/%Y %I:%M:%S.%f %p",    # with millis
    "%m/%d/%Y %H:%M:%S",          # 24h
    "%m/%d/%Y %H:%M:%S.%f",       # 24h + millis
    "%Y-%m-%d %H:%M:%S",          # ISO-ish
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
]

_TZ_SUFFIX_RE = re.compile(r"(Z|[+-]\d{2}:\d{2})$")

def _parse_fitbit_time_series(series: pd.Series) -> pd.Series:
    """
    Deterministic parsing for Fitbit Time column:
    - Strips 'Z' / timezone offsets to keep everything naive
    - Attempts a known list of formats in order
    - Localizes final result to UTC (no shift)
    """
    s = series.astype(str).str.strip()

    # Normalize: drop trailing timezone markers so we don't create mixed tz-aware/naive series
    s = s.str.replace(_TZ_SUFFIX_RE, "", regex=True)

    ts = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    for fmt in _FITBIT_FORMATS:
        mask = ts.isna()
        if not mask.any():
            break
        parsed = pd.to_datetime(s[mask], format=fmt, errors="coerce")
        ts.loc[mask] = parsed

    # If anything still NaT, try a quiet, last-resort parse (no warnings shown)
    remaining = ts.isna()
    if remaining.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts.loc[remaining] = pd.to_datetime(s[remaining], errors="coerce")

    # Drop rows that still failed
    return ts

def _clean_df(ts: pd.Series, hr: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": ts, "hr": pd.to_numeric(hr, errors="coerce")}).dropna()

    # Localize to UTC (no shift) for consistency in downstream code/UI
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize("UTC")
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert("UTC")

    # Keep physiologically plausible values
    out = out[(out["hr"] >= 25) & (out["hr"] <= 240)]
    if out.empty:
        raise HTTPException(400, "No valid rows after parsing.")
    return out.sort_values("timestamp").reset_index(drop=True)

def _read_all_users(fobj):
    """
    Returns:
      users: list of dicts {id: <str|None>, df: DataFrame[timestamp,hr] sorted by time}
      ids:   list of user Ids (empty if generic schema)
    Supports:
      - Generic: timestamp,hr
      - Fitbit Kaggle: Id,Time,Value
    """
    df = pd.read_csv(fobj, low_memory=False)
    cols = {c.lower(): c for c in df.columns}

    users, ids = [], []

    # Simple schema
    if "timestamp" in cols and "hr" in cols:
        ts = pd.to_datetime(df[cols["timestamp"]], errors="coerce", utc=True)
        users.append({"id": None, "df": _clean_df(ts, df[cols["hr"]])})
        return users, ids

    # Fitbit schema
    if "time" in cols and "value" in cols:
        ts_all = _parse_fitbit_time_series(df[cols["time"]])

        if "id" in cols:
            id_col = cols["id"]
            # Ensure stable per-user ordering by time
            for uid, g in df.groupby(id_col, sort=False):
                ids.append(str(uid))
                mask = (df[id_col] == uid)
                ts = ts_all[mask]
                users.append({"id": str(uid), "df": _clean_df(ts, df.loc[mask, cols["value"]])})
        else:
            users.append({"id": None, "df": _clean_df(ts_all, df[cols["value"]])})
        return users, ids

    raise HTTPException(
        400,
        f"CSV must have ('timestamp','hr') or Fitbit ('Id','Time','Value'). Found: {list(df.columns)}"
    )

# -------------------- Prediction ------------------
def _predict_single(df: pd.DataFrame, win_sec: int = 60, step_sec: int = 10):
    # 1 Hz resample → guarantees correct, continuous timeline
    s = df.set_index("timestamp")["hr"].resample("1s").mean().interpolate(limit_direction="both")

    x_seq, x_feat, t_idx = to_windows(s.values.astype("float32"), s.index, win_sec, step_sec)
    if len(x_seq) == 0:
        raise HTTPException(400, "Not enough data for 60-second windows.")

    with torch.no_grad():
        P_cnn = torch.softmax(cnn(torch.from_numpy(x_seq[:, None, :]).float()), dim=1).numpy()
    P_svm = svm.predict_proba(x_feat)

    # simple rule prior
    priors = []
    for z in x_seq:
        std = float(np.std(z)); mean = float(np.mean(z))
        p = np.array([0.25,0.25,0.25,0.25], np.float32)
        if mean > 1.0:  p = np.array([0.10,0.70,0.10,0.10], np.float32)
        if mean < -1.0: p = np.array([0.10,0.10,0.70,0.10], np.float32)
        if std > 1.2:   p = np.array([0.15,0.15,0.10,0.60], np.float32)
        priors.append(p)
    P_rule = np.stack(priors)

    Z = np.hstack([P_svm, P_cnn, P_rule])
    p_meta = meta.predict_proba(Z)
    y_pred = p_meta.argmax(1); conf = p_meta.max(1)

    # merge contiguous windows into episodes
    episodes = []
    if len(y_pred):
        cur_label = y_pred[0]; cur_start = t_idx[0]; cur_conf = [conf[0]]
        for i in range(1, len(y_pred)):
            if y_pred[i] == cur_label and (t_idx[i]-t_idx[i-1]).total_seconds() <= step_sec + 1:
                cur_conf.append(conf[i])
            else:
                episodes.append({
                    "label": CLASSES[int(cur_label)],
                    "start": cur_start.isoformat(),
                    "end": (t_idx[i-1] + pd.Timedelta(seconds=win_sec)).isoformat(),
                    "confidence": float(np.mean(cur_conf)),
                })
                cur_label = y_pred[i]; cur_start = t_idx[i]; cur_conf = [conf[i]]
        episodes.append({
            "label": CLASSES[int(cur_label)],
            "start": cur_start.isoformat(),
            "end": (t_idx[-1] + pd.Timedelta(seconds=win_sec)).isoformat(),
            "confidence": float(np.mean(cur_conf)),
        })

    # downsample for plotting if huge, keep chronological order
    series_df = df
    max_pts = 20000
    if len(series_df) > max_pts:
        step = int(np.ceil(len(series_df) / max_pts))
        series_df = series_df.iloc[::step, :]

    series = [{"t": ts.isoformat(), "hr": float(v)} for ts, v in zip(series_df["timestamp"], series_df["hr"])]
    return {"episodes": episodes, "series": series}

# -------------------- Routes ----------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    user_id: str | None = Form(default=None),
    multi_users: bool = Form(default=False),
):
    tmp_path = _save_stream_to_tmp(file)
    try:
        with _open_any_csv(tmp_path, file.filename or "upload.csv") as fobj:
            users, ids = _read_all_users(fobj)

        # Filter to a specific user if requested
        if user_id is not None:
            users = [u for u in users if u["id"] == user_id]
            if not users:
                raise HTTPException(400, f"user_id {user_id} not found in CSV")

        if multi_users:
            results = []
            for u in users:
                res = _predict_single(u["df"])
                results.append({"id": u["id"], **res})
            return {"multi": True, "ids": ids, "results": results}
        else:
            u = users[0]
            res = _predict_single(u["df"])
            payload = {"multi": False, **res}
            if ids:
                payload["ids"] = ids
                payload["selected_id"] = u["id"]
            return payload
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
