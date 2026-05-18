# HospitalLOS — Prediksi Lama Tinggal Pasien Rumah Sakit

> **UTS Praktikum Kecerdasan Buatan** · Semester 4 (Genap) 
> Program Studi Teknik Informatika · Fakultas Teknologi Informasi

---

##  Identitas Mahasiswa

| | |
|---|---|
| **Nama** | Selsa Shafana Alfiyani |
| **NIM** | 301240041 |
| **Kelas** | 4B |
| **Mata Kuliah** | Praktikum Kecerdasan Buatan |
| **Topik** | Prediksi Lama Tinggal Pasien RS (No. 21) |

---

##  Deskripsi Proyek

HospitalLOS adalah sistem prediksi lama rawat inap pasien rumah sakit berbasis machine learning yang dibangun sebagai tugas UTS Praktikum Kecerdasan Buatan. Sistem ini mengimplementasikan 5 algoritma machine learning secara komparatif untuk memperkirakan berapa hari seorang pasien akan dirawat sejak hari pertama masuk.

Sistem ini bertujuan membantu tenaga medis dan manajemen rumah sakit dalam merencanakan kapasitas tempat tidur dan alokasi sumber daya secara lebih efisien berdasarkan kondisi klinis pasien.

---

##  Algoritma yang Diimplementasikan

| No | Algoritma | Library | Metrik Evaluasi |
|---|---|---|---|
| 1 | **Linear Regression** | `scikit-learn` | MAE, RMSE, R² |
| 2 | **Artificial Neural Network (ANN)** | `TensorFlow / Keras` | MAE, RMSE, Loss Curve |
| 3 | **LSTM (Long Short-Term Memory)** | `TensorFlow / Keras` | MAE, RMSE, MAPE |
| 4 | **K-Means Clustering** | `scikit-learn` | Inertia, Silhouette Score |
| 5 | **Backpropagation (Manual)** | `NumPy` | Loss, Convergence |

---

##  Dataset

- **Nama**: Hospital Length of Stay Dataset — Microsoft
- **Sumber**: [Kaggle](https://www.kaggle.com/datasets/aayushchou/hospital-length-of-stay-dataset-microsoft)
- **Lisensi**: CC BY 4.0
- **Jumlah Baris**: 100.000
- **Jumlah Kolom**: 28
- **Target Variabel**: `lengthofstay` (1–17 hari, rata-rata 4 hari)

### Fitur Utama Dataset

| Fitur | Deskripsi | Tipe |
|---|---|---|
| `rcount` | Jumlah kunjungan rawat inap sebelumnya | Kategorikal |
| `gender` | Jenis kelamin (F/M) | Kategorikal |
| `facid` | Kode fasilitas RS (A–E) | Kategorikal |
| `hematocrit` | Kadar sel darah merah | Numerik |
| `neutrophils` | Jumlah sel darah putih neutrofil | Numerik |
| `sodium` | Kadar natrium darah | Numerik |
| `glucose` | Kadar gula darah | Numerik |
| `bloodureanitro` | Blood Urea Nitrogen (fungsi ginjal) | Numerik |
| `creatinine` | Kreatinin (fungsi ginjal) | Numerik |
| `bmi` | Indeks massa tubuh | Numerik |
| `pulse` | Detak jantung per menit | Numerik |
| `respiration` | Laju pernapasan | Numerik |
| `dialysisrenalendstage` – `secondarydiagnosisnonicd9` | 12 fitur penyakit penyerta (biner) | Numerik |
| `lengthofstay` | **Target** — lama rawat inap (hari) | **Target** |

---

##Fitur Aplikasi Web

- **Mode Pasien** — input kondisi umum tanpa data lab, cocok untuk keluarga pasien
- **Mode Tenaga Medis** — input nilai lab lengkap untuk prediksi akurat
- **Upload CSV** — prediksi batch banyak pasien sekaligus, hasil bisa diunduh
- **Data Contoh** — ambil data acak dari dataset asli sebagai demonstrasi
- **Perbandingan Model** — grafik interaktif RMSE, R², loss curve, elbow method
- **Segmentasi K-Means** — pasien dikategorikan Rawat Singkat / Sedang / Lama
- **Saran Klinis** — rekomendasi otomatis berdasarkan nilai input
- **Dark Mode** — toggle tema terang/gelap
- **Responsif** — tampilan optimal di desktop dan mobile

---

## Struktur Folder

```
hospital-los-predict/
│
├── data/
│   └── LengthOfStay.csv          # Dataset mentah (100.000 baris)
│
├── models/                        # File model tersimpan (di-generate saat training)
│   ├── linear_regression.pkl
│   ├── ann_model.keras
│   ├── lstm_model.keras
│   ├── kmeans_model.pkl
│   ├── kmeans_cluster_label_map.pkl
│   ├── backprop_model.pkl
│   ├── scaler.pkl
│   ├── encoding_maps.pkl
│   ├── feature_names.pkl
│   ├── data_split.pkl
│   ├── metrics_*.pkl
│   ├── *_history.pkl
│   └── kmeans_elbow_data.pkl
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── img/                       # Grafik EDA & model (di-generate saat training)
│
├── templates/
│   ├── home.html
│   ├── predict.html
│   ├── compare.html
│   ├── dataset.html
│   └── about.html
│
├── train/
│   ├── 01_EDA.py                  # Exploratory Data Analysis
│   ├── 02_preprocessing.py        # Preprocessing & split data
│   ├── 03_linear_regression.py    # Training Linear Regression
│   ├── 04_ann.py                  # Training ANN
│   ├── 05_lstm.py                 # Training LSTM
│   ├── 06_kmeans.py               # Training K-Means Clustering
│   └── 07_backpropagation.py      # Training Backpropagation manual
│
├── app.py                         # Flask web application (entry point)
├── Procfile                       # Konfigurasi deploy Railway/Render
├── requirements.txt               # Dependensi Python
└── README.md
```

---

##  Cara Instalasi & Menjalankan Lokal

### 1. Clone Repository

```bash
git clone https://github.com/username/hospital-los-predict.git
cd hospital-los-predict
```

### 2. Buat Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Siapkan Dataset

Letakkan file `LengthOfStay.csv` di folder `data/`:

```
data/
└── LengthOfStay.csv
```

Download dari: [Kaggle — Hospital Length of Stay](https://www.kaggle.com/datasets/aayushchou/hospital-length-of-stay-dataset-microsoft)

### 5. Jalankan Training Model (urut)

```bash
python train/01_EDA.py
python train/02_preprocessing.py
python train/03_linear_regression.py
python train/04_ann.py
python train/05_lstm.py
python train/06_kmeans.py
python train/07_backpropagation.py
```

Setelah selesai, folder `models/` dan `static/img/` akan terisi otomatis.

### 6. Jalankan Aplikasi Web

```bash
python app.py
```

Buka browser: [http://localhost:5000](http://localhost:5000)

---

### Konfigurasi `Procfile`

```
web: gunicorn app:app
```

##  Dependencies

```
flask==3.0.3
werkzeug==3.0.3
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.0
scipy==1.13.1
threadpoolctl==3.5.0
tensorflow==2.15.0
keras==2.15.0
matplotlib==3.9.0
seaborn==0.13.2
joblib==1.4.2
gunicorn==22.0.0
```

---

##  Tech Stack

| Kategori | Teknologi |
|---|---|
| **Backend** | Python 3.11, Flask 3.0 |
| **Machine Learning** | scikit-learn, TensorFlow/Keras, NumPy |
| **Data Processing** | Pandas, NumPy |
| **Visualisasi** | Matplotlib, Seaborn, Chart.js |
| **Frontend** | Bootstrap 5, AOS.js, Bootstrap Icons |
| **Deployment** | Railway, Gunicorn |
| **Domain** | Cloudflare DNS + `.my.id` |

---

##  Alur Kerja Penelitian

```
Dataset (100.000 baris)
        │
        ▼
01_EDA.py ──────────── Eksplorasi data, distribusi, korelasi, visualisasi
        │
        ▼
02_preprocessing.py ── Hapus kolom tidak relevan, encoding, normalisasi,
                        split 70% train / 15% val / 15% test
        │
        ├──▶ 03_linear_regression.py ── Baseline model
        ├──▶ 04_ann.py ──────────────── Neural network multilayer
        ├──▶ 05_lstm.py ─────────────── Sequential model (sliding window)
        ├──▶ 06_kmeans.py ───────────── Clustering & segmentasi pasien
        └──▶ 07_backpropagation.py ──── Neural network manual NumPy
                        │
                        ▼
                  models/ tersimpan
                        │
                        ▼
                    app.py (Flask)
                        │
                        ▼
              Aplikasi Web HospitalLOS
```

---

##  Navigasi Aplikasi

| Halaman | URL | Deskripsi |
|---|---|---|
| Home | `/` | Landing page, ringkasan metrik model terbaik |
| Prediksi | `/predict` | Input data pasien, hasil prediksi real-time |
| Perbandingan Model | `/compare` | Grafik interaktif perbandingan 5 algoritma |
| Dataset | `/dataset` | Statistik deskriptif & preview dataset |
| Tentang | `/about` | Identitas mahasiswa & informasi proyek |

---

##  Catatan Penting

- Hasil prediksi bersifat **estimasi** dan **tidak menggantikan** keputusan tenaga medis profesional
- Model dilatih menggunakan dataset Microsoft Hospital LOS; performa dapat berbeda pada data dari rumah sakit yang berbeda
- Nilai input pada Mode Tenaga Medis mengikuti **distribusi dataset pelatihan**, bukan selalu sama dengan rentang klinis standar internasional

---

##  Lisensi

Proyek ini dibuat untuk keperluan akademik (UTS Praktikum Kecerdasan Buatan).  
Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Microsoft via Kaggle.

---

<div align="center">
  <p>Dibuat oleh <strong>Selsa Shafana Alfiyani</strong> · 301240041 · 4B</p>
  <p>UTS Praktikum Kecerdasan Buatan · Teknik Informatika · 2026</p>
</div>
