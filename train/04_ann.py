import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH = "models"
IMG_PATH   = "static/img"
os.makedirs(IMG_PATH, exist_ok=True)
EPOCHS     = 100
BATCH_SIZE = 256

plt.rcParams["figure.dpi"]        = 150
plt.rcParams["font.family"]       = "sans-serif"
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# ============================================================
# MULAI
# ============================================================
print("=" * 55)
print("  04 ANN — Prediksi Lama Tinggal Pasien RS")
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
y_train = np.array(y_train)
y_val   = np.array(y_val)
y_test  = np.array(y_test)
n_features = X_train.shape[1]
print(f"      Train  : {len(X_train):,} baris")
print(f"      Val    : {len(X_val):,} baris")
print(f"      Test   : {len(X_test):,} baris")
print(f"      Fitur  : {n_features}")
print()

# ============================================================
# ARSITEKTUR MODEL
# ============================================================
print("[2/5] Membangun arsitektur ANN...")
model = Sequential([
    Dense(128, activation="relu", input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64,  activation="relu"),
    Dropout(0.2),
    Dense(32,  activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])
model.summary()
print()

# ============================================================
# TRAINING
# ============================================================
print("[3/5] Melatih model ANN...")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, verbose=1)
]
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
print()

# ============================================================
# EVALUASI
# ============================================================
print("[4/5] Evaluasi model...")
y_pred_test = model.predict(X_test, verbose=0).flatten()
y_pred_val  = model.predict(X_val,  verbose=0).flatten()

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
joblib.dump(metrics, f"{MODEL_PATH}/metrics_ann.pkl")
print("      Metrik tersimpan : models/metrics_ann.pkl")

model.save(f"{MODEL_PATH}/ann_model.h5")
print("      Model tersimpan  : models/ann_model.keras")

joblib.dump(history.history, f"{MODEL_PATH}/ann_history.pkl")
print("      History tersimpan: models/ann_history.pkl")
print()

# ============================================================
# GRAFIK
# ============================================================
print("[5/5] Membuat grafik...")
print("      Grafik 1/1: Loss curve ANN...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history["loss"],     color="#2563EB", linewidth=1.5, label="Train Loss")
ax.plot(history.history["val_loss"], color="#EF4444", linewidth=1.5, label="Val Loss", linestyle="--")
ax.set_title("ANN — Training & Validation Loss", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE)")
ax.legend()
ax.text(0.98, 0.95,
        f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}",
        transform=ax.transAxes, fontsize=9, ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/ann_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: ann_loss_curve.png")
print()

# ============================================================
# SELESAI
# ============================================================
print("=" * 55)
print(f"  ANN selesai.")
print(f"  MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
print("=" * 55)