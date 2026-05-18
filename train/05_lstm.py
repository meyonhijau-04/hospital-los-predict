import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH  = "models"
IMG_PATH    = "static/img"
os.makedirs(IMG_PATH, exist_ok=True)
EPOCHS      = 100
BATCH_SIZE  = 256
WINDOW_SIZE = 5

plt.rcParams["figure.dpi"]        = 150
plt.rcParams["font.family"]       = "sans-serif"
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# ============================================================
# MULAI
# ============================================================
print("=" * 55)
print("  05 LSTM — Prediksi Lama Tinggal Pasien RS")
print("=" * 55)
print()

# ============================================================
# LOAD DATA
# ============================================================
print("[1/6] Memuat data hasil preprocessing...")
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
# RESHAPE UNTUK LSTM (setiap baris = 1 timestep)
# ============================================================
print(f"[2/6] Reshape data untuk LSTM (timestep=1)...")
X_train_seq = X_train.reshape((X_train.shape[0], 1, n_features))
X_val_seq   = X_val.reshape((X_val.shape[0],   1, n_features))
X_test_seq  = X_test.reshape((X_test.shape[0],  1, n_features))
print(f"      Shape train : {X_train_seq.shape}")
print(f"      Shape val   : {X_val_seq.shape}")
print(f"      Shape test  : {X_test_seq.shape}")
print()

# ============================================================
# ARSITEKTUR MODEL
# ============================================================
print("[3/6] Membangun arsitektur LSTM...")
model = Sequential([
    LSTM(128, return_sequences=True,
         input_shape=(1, n_features)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])
model.summary()
print()

# ============================================================
# TRAINING
# ============================================================
print("[4/6] Melatih model LSTM...")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=7, verbose=1)
]
history = model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
print()

# ============================================================
# EVALUASI
# ============================================================
print("[5/6] Evaluasi model...")
y_pred_test = model.predict(X_test_seq, verbose=0).flatten()
y_pred_val  = model.predict(X_val_seq,  verbose=0).flatten()

mae  = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2   = r2_score(y_test, y_pred_test)
mape = np.mean(np.abs((y_test - y_pred_test) /
               np.where(y_test == 0, 1, y_test))) * 100

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
print(f"      MAPE             : {mape:.2f}%")
print()

metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
joblib.dump(metrics, f"{MODEL_PATH}/metrics_lstm.pkl")
print("      Metrik tersimpan : models/metrics_lstm.pkl")

model.save(f"{MODEL_PATH}/lstm_model.h5")
print("      Model tersimpan  : models/lstm_model.keras")

joblib.dump(history.history, f"{MODEL_PATH}/lstm_history.pkl")
print("      History tersimpan: models/lstm_history.pkl")

WINDOW_SIZE_SAVE = 1
joblib.dump(WINDOW_SIZE_SAVE, f"{MODEL_PATH}/lstm_window_size.pkl")
print(f"      Window size      : {WINDOW_SIZE_SAVE}")
print()

# ============================================================
# GRAFIK
# ============================================================
print("[6/6] Membuat grafik...")
print("      Grafik 1/2: Loss curve LSTM...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history["loss"],     color="#2563EB", linewidth=1.5, label="Train Loss")
ax.plot(history.history["val_loss"], color="#EF4444", linewidth=1.5, label="Val Loss", linestyle="--")
ax.set_title("LSTM — Training & Validation Loss", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE)")
ax.legend()
ax.text(0.98, 0.95,
        f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}",
        transform=ax.transAxes, fontsize=9, ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/lstm_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: lstm_loss_curve.png")

print("      Grafik 2/2: Actual vs Predicted LSTM...")
sample_idx = np.random.choice(len(y_test), size=200, replace=False)
sample_sort = np.argsort(sample_idx)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(200), y_test[sample_idx][sample_sort],
        color="#2563EB", linewidth=1.2, label="Actual")
ax.plot(range(200), y_pred_test[sample_idx][sample_sort],
        color="#EF4444", linewidth=1.2, label="Predicted", linestyle="--")
ax.set_title("LSTM — Actual vs Predicted (200 sampel)", fontsize=12, fontweight="bold")
ax.set_xlabel("Sampel")
ax.set_ylabel("Lama Rawat (Hari)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG_PATH}/lstm_actual_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Tersimpan: lstm_actual_vs_pred.png")
print()

# ============================================================
# SELESAI
# ============================================================
print("=" * 55)
print(f"  LSTM selesai.")
print(f"  MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | MAPE={mape:.2f}%")
print("=" * 55)