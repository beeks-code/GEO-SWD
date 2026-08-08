import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.transform import train_transform, sliced_wasserstein


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print("Device:", device)


# load and split ds
data = load_breast_cancer()

X = data.data.astype(np.float32)
y = data.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)


# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)

## Obtain two dist
A = X_train_t[y_train_t == 0]
B = X_train_t[y_train_t == 1]

print("\n")
print("Training distribution")
print("Class 0:", A.shape)
print("Class 1:", B.shape)


# Original distribution separation
with torch.no_grad():
    original_swd = sliced_wasserstein(A, B, num_projections=1024)

print("\n")
print("ORIGINAL DISTRIBUTION")
print(f"Original SWD: {original_swd.item():.6f}")


# Train SWD-only transformation
print("\n" + "=" * 75)
print("TRAINING SWD-ONLY TRANSFORMATION")
print("=" * 75)

T_swd = train_transform(A, B, use_geometry=False, epochs=2000, lr=1e-3, lambda_structure=0.0, num_projections=128, num_pairs=3000)


# Train SWD + geometry transformation
print("\n" + "=" * 75)
print("TRAINING SWD + GEOMETRY TRANSFORMATION")
print("=" * 75)

T_geometry = train_transform(A, B, use_geometry=True, epochs=2000, lr=1e-3, lambda_structure=0.1, num_projections=128, num_pairs=3000)


# Transform training and test data
with torch.no_grad():
    X_train_swd = T_swd(X_train_t)
    X_test_swd = T_swd(X_test_t)
    X_train_geometry = T_geometry(X_train_t)
    X_test_geometry = T_geometry(X_test_t)


# Measure final distribution separation
A_swd = X_train_swd[y_train_t == 0]
B_swd = X_train_swd[y_train_t == 1]

A_geometry = X_train_geometry[y_train_t == 0]
B_geometry = X_train_geometry[y_train_t == 1]

with torch.no_grad():
    swd_only = sliced_wasserstein(A_swd, B_swd, num_projections=2048)
    swd_geometry = sliced_wasserstein(A_geometry, B_geometry, num_projections=2048)

print("\n" + "=" * 75)
print("FINAL DISTRIBUTION SEPARATION")
print("=" * 75)
print(f"Original       : {original_swd.item():.6f}")
print(f"SWD-only       : {swd_only.item():.6f}")
print(f"SWD + Geometry : {swd_geometry.item():.6f}")


results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)

output_file = results_dir / "decision_tree_swd_experiment.pt"

torch.save({
    "X_train_original": X_train_t.cpu(),
    "X_test_original": X_test_t.cpu(),
    "X_train_swd": X_train_swd.cpu(),
    "X_test_swd": X_test_swd.cpu(),
    "X_train_geometry": X_train_geometry.cpu(),
    "X_test_geometry": X_test_geometry.cpu(),
    "y_train": torch.tensor(y_train),
    "y_test": torch.tensor(y_test),

    "original_swd": original_swd.item(),
    "swd_only": swd_only.item(),
    "swd_geometry": swd_geometry.item(),

    "transform_swd_state_dict": T_swd.state_dict(),
    "transform_geometry_state_dict": T_geometry.state_dict(),

    "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
    "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32),

    "feature_names": data.feature_names.tolist(),
    "target_names": data.target_names.tolist()
}, output_file)

print("\n" + "=" * 75)
print("SAVED")
print("=" * 75)
print(output_file)


from decision_tree import run_decision_tree_experiment

run_decision_tree_experiment(
    X_train, X_test,
    X_train_swd.cpu().numpy(), X_test_swd.cpu().numpy(),
    X_train_geometry.cpu().numpy(), X_test_geometry.cpu().numpy(),
    y_train, y_test,
    output_file
)
