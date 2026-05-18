import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# FIX: import BackpropNetwork dari module terpisah
# agar pkl tersimpan dengan referensi backprop_model_class.BackpropNetwork
# bukan __main__.BackpropNetwork yang tidak bisa diload oleh Gunicorn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backprop_model_class import BackpropNetwork

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH  = "models"
IMG_PATH    = "static/img"
os.makedirs(IMG_PATH, exist_ok=True)
EPOCHS      = 200
LR          = 0.01
BATCH_SIZE  = 256
HIDDEN1     = 64
HIDDEN2     = 32

plt.rcParams["figure.dpi"]        = 150
plt.rcParams["font.family"]       = "sans-serif"
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ============================================================
# MULAI
# ============================================================
print("=" * 55)
print("  07 BACKPROPAGATION — Prediksi Lama Tinggal Pasien RS")
print("=" * 55)
print()

# ============================================================
# LOAD DATA
# ============================================================
print("[1/5] Memuat data hasil preprocessing...")
X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
    f"{MODEL_PATH}/data_split.pkl"
)
X_train = X_train.values
X_val   = X_val.values
X_test  = X_test.values
y_train = np.array(y_train, dtype=float)
y_val   = np.array(y_val,   dtype=float)
y_test  = np.array(y_test,  dtype=float)
n_features = X_train.shape[1]
print(f"      Train  : {len(X_train):,} baris")
print(f"      Val    : {len(X_val):,} baris")
print(f"      Test   : {len(X_test):,} baris")
print(f"      Fitur  : {n_features}")
print()

# ============================================================
# INISIALISASI MODEL
# ============================================================
print("[2/5] Inisialisasi Backpropagation Network...")
print(f"      Arsitektur       : {n_features} -> {HIDDEN1} -> {HIDDEN2} -> 1")
print(f"      Learning rate    : {LR}")
print(f"      Batch size       : {BATCH_SIZE}")
print(f"      Max epochs       : {EPOCHS}")
print()
model = BackpropNetwork(n_features, HIDDEN1, HIDDEN2, lr=LR)

# ============================================================
# TRAINING
# ============================================================
print("[3/5] Melatih model Backpropagation...")
train_losses  = []
val_losses    = []
best_val_loss = np.inf
patience      = 15
no_improve    = 0
best_weights  = None

for epoch in range(1, EPOCHS + 1):
    idx    = np.random.permutation(len(X_train))
    X_shuf = X_train[idx]
    y_shuf = y_train[idx]

    for i in range(0, len(X_shuf), BATCH_SIZE):
        Xb = X_shuf[i:i+BATCH_SIZE]
        yb = y_shuf[i:i+BATCH_SIZE]
        yp = model.forward(Xb)
        model.backward(Xb, yb, yp)

    y_pred_tr = model.predict(X_train)
    y_pred_vl = model.predict(X_val)
    loss_tr   = mse_loss(y_train, y_pred_tr)
    loss_vl   = mse_loss(y_val,   y_pred_vl)
    train_losses.append(loss_tr)
    val_losses.append(loss_vl)

    if epoch % 20 == 0 or epoch == 1:
        print(f"      Epoch {epoch:4d}/{EPOCHS}  "
              f"Train Loss={loss_tr:.4f}  Val Loss={loss_vl:.4f}")

    if loss_vl < best_val_loss:
        best_val_loss = loss_vl
        best_weights  = (model.W1.copy(), model.b1.copy(),
                         model.W2.copy(), model.b2.copy(),
                         model.W3.copy(), model.b3.copy())
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"      Early stopping di epoch {epoch}")
            break

if best_weights:
    (model.W1, model.b1,
     model.W2, model.b2,
     model.W3, model.b3) = best_weights
print()

# ============================================================
# EVALUASI
# ============================================================
print("[4/5] Evaluasi model...")
y_pred_test = model.predict(X_test)
y_pred_val  = model.predict(X_val)

mae  = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2   = r2_score(y_test, y_pred_test)

mae_val  = mean_absolute_error(y_val, y_pred_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val   = r2_score(y_val, y_pred_val)

print(f"      --- Validasi ---")
print(f"      MAE              : {mae_val:.4f}")
print(f"      RMSE             : {rmse_val:.4f}")
print(f"      R²               : {r2_val:.4f}")
print()
print(f"      --- Test ---")
print(f"      MAE              : {mae:.4f}")
print(f"      RMSE             : {rmse:.4f}")
print(f"      R²               : {r2:.4f}")
print()

metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
joblib.dump(metrics, f"{MODEL_PATH}/metrics_backprop.pkl")
joblib.dump(model,   f"{MODEL_PATH}/backprop_model.pkl")
print("      Metrik tersimpan : models/metrics_backprop.pkl")
print("      Model tersimpan  : models/backprop_model.pkl")
print()

# ============================================================
# GRAFIK
# ============================================================
print("[5/5] Membuat grafik perbandingan semua model...")

print("      Grafik 1/3: Loss curve Backpropagation...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, color="#2563EB", linewidth=1.5, label="Train Loss")
ax.plot(val_losses,   color="#EF4444", linewidth=1.5, label="Val Loss", linestyle="--")
ax.set_title("Backpropagation — Convergence Loss Curve", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE)")
ax.legend()
ax.text(0.98, 0.95,
        f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}",
        transform=ax.transAxes, fontsize=9, ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/backprop_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: backprop_loss_curve.png")

print("      Grafik 2/3: Perbandingan RMSE semua model...")
m_lr  = joblib.load(f"{MODEL_PATH}/metrics_linear_regression.pkl")
m_ann = joblib.load(f"{MODEL_PATH}/metrics_ann.pkl")
m_lst = joblib.load(f"{MODEL_PATH}/metrics_lstm.pkl")
m_km  = joblib.load(f"{MODEL_PATH}/metrics_kmeans.pkl")
m_bp  = metrics

model_names = ["Linear\nRegression", "ANN", "LSTM", "Backprop"]
rmse_vals   = [m_lr["RMSE"], m_ann["RMSE"], m_lst["RMSE"], m_bp["RMSE"]]
colors_bar  = ["#93C5FD", "#2563EB", "#1D4ED8", "#60A5FA"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(model_names, rmse_vals, color=colors_bar, alpha=0.85, width=0.5)
ax.set_title("Perbandingan RMSE Semua Model", fontsize=12, fontweight="bold")
ax.set_ylabel("RMSE")
for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/model_comparison_rmse.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: model_comparison_rmse.png")

print("      Grafik 3/3: Perbandingan R² semua model...")
r2_vals = [m_lr["R2"], m_ann["R2"], m_lst["R2"], m_bp["R2"]]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(model_names, r2_vals, color=colors_bar, alpha=0.85, width=0.5)
ax.set_title("Perbandingan R² Semua Model", fontsize=12, fontweight="bold")
ax.set_ylabel("R² Score")
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/model_comparison_r2.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: model_comparison_r2.png")
print()

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print("=" * 55)
print("  RINGKASAN SEMUA MODEL")
print("=" * 55)
print(f"  {'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print(f"  {'-'*46}")
print(f"  {'Linear Regression':<20} {m_lr['MAE']:>8.4f} {m_lr['RMSE']:>8.4f} {m_lr['R2']:>8.4f}")
print(f"  {'ANN':<20} {m_ann['MAE']:>8.4f} {m_ann['RMSE']:>8.4f} {m_ann['R2']:>8.4f}")
print(f"  {'LSTM':<20} {m_lst['MAE']:>8.4f} {m_lst['RMSE']:>8.4f} {m_lst['R2']:>8.4f}")
print(f"  {'Backpropagation':<20} {m_bp['MAE']:>8.4f} {m_bp['RMSE']:>8.4f} {m_bp['R2']:>8.4f}")
print(f"  {'K-Means':<20} {'—':>8} {'—':>8} {m_km['silhouette']:>8.4f}*")
print(f"  {'-'*46}")
print(f"  *K-Means menggunakan Silhouette Score")
print()

all_rmse = {"Linear Regression": m_lr["RMSE"], "ANN": m_ann["RMSE"],
            "LSTM": m_lst["RMSE"], "Backpropagation": m_bp["RMSE"]}
best_model = min(all_rmse, key=all_rmse.get)
print(f"  Model terbaik    : {best_model} (RMSE={all_rmse[best_model]:.4f})")
print("=" * 55)