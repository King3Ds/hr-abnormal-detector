"""
Trains tiny synthetic SVM + 1D-CNN models and a fusion meta-learner.
Creates:
  models/svm.joblib
  models/cnn.pt
  models/fuser.joblib
Also writes a small sample CSV: sample_heartrate.csv
"""
from pathlib import Path
import sys, numpy as np, joblib, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch, torch.nn as nn, torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.synth import make_dataset, CLASSES
from src.models.cnn import HR1DCNN

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True, parents=True)

# 1) data
X_seq, X_feat, y = make_dataset(n_per_class=200, T=60, seed=7)
X_seq_tr, X_seq_va, X_feat_tr, X_feat_va, y_tr, y_va = train_test_split(
    X_seq, X_feat, y, test_size=0.2, stratify=y, random_state=0
)

# 2) SVM (calibrated)
svm_pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', class_weight='balanced', probability=True))])
svm_calib = CalibratedClassifierCV(svm_pipe, cv=3, method='sigmoid')
svm_calib.fit(X_feat_tr, y_tr)
joblib.dump(svm_calib, MODELS / "svm.joblib")

# 3) CNN (quick CPU training)
device = torch.device("cpu")
model = HR1DCNN(n_classes=len(CLASSES)).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

def batches(X, Y, bs=64):
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    for i in range(0, n, bs):
        j = idx[i:i+bs]
        yield torch.from_numpy(X[j][:, None, :]).float(), torch.from_numpy(Y[j]).long()

for epoch in range(8):
    model.train()
    for xb, yb in batches(X_seq_tr, y_tr, 64):
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()

torch.save(model.state_dict(), MODELS / "cnn.pt")

# 4) fusion meta-learner (on validation probs + simple rule priors)
with torch.no_grad():
    P_cnn_va = torch.softmax(model(torch.from_numpy(X_seq_va[:, None, :]).float()), dim=1).numpy()
P_svm_va = svm_calib.predict_proba(X_feat_va)

def rule_prior_from_zseq(batch):
    out = []
    for z in batch:
        std = float(np.std(z))
        mean = float(np.mean(z))
        p = np.array([0.25,0.25,0.25,0.25], dtype=np.float32)
        if mean > 1.0:  p = np.array([0.1,0.7,0.1,0.1], dtype=np.float32)
        if mean < -1.0: p = np.array([0.1,0.1,0.7,0.1], dtype=np.float32)
        if std > 1.2:   p = np.array([0.15,0.15,0.1,0.6], dtype=np.float32)
        out.append(p)
    return np.stack(out)

P_rule_va = rule_prior_from_zseq(X_seq_va)
Z_va = np.hstack([P_svm_va, P_cnn_va, P_rule_va])
meta = LogisticRegression(max_iter=200).fit(Z_va, y_va)
joblib.dump(meta, MODELS / "fuser.joblib")

# 5) sample CSV (20 minutes: normal → tachy → brady → irregular)
rng = np.random.default_rng(0)
def seg(kind, n):
    if kind=='tachy': return rng.normal(130,4,n)
    if kind=='brady': return rng.normal(40,3,n)
    if kind=='irregular':
        x=rng.normal(70,3,n); 
        for i in range(0,n,7): x[i:i+3]+=rng.normal(0,18,min(3,n-i))
        return x
    return rng.normal(70,3,n)

import pandas as pd
t0 = pd.Timestamp("2025-01-01T00:00:00Z")
hr = np.concatenate([seg('normal',300), seg('tachy',300), seg('brady',300), seg('irregular',300)]).astype('float32')
ts = pd.date_range(t0, periods=len(hr), freq='1S')
pd.DataFrame({'timestamp': ts, 'hr': hr}).to_csv(ROOT / "sample_heartrate.csv", index=False)

print("✅ Trained and saved models to ./models and wrote sample_heartrate.csv")