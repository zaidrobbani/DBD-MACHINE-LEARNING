# MLOps-DBD

Fondasi teknis proyek MLOps untuk membangun model *machine learning* prediksi/klasifikasi **DBD (Demam Berdarah Dengue)** berbasis data tabular. Proyek ini dibangun agar **reproducible** (dapat dijalankan ulang di lingkungan mana pun melalui GitHub Codespaces) dan dikelola dengan strategi branching **GitHub Flow**.

## Tujuan Proyek

Membangun pipeline machine learning end-to-end (mulai dari eksplorasi data, feature engineering, training, hingga evaluasi model) untuk memprediksi/mengklasifikasikan kasus DBD berdasarkan data yang tersedia, dengan lingkungan kerja yang konsisten dan terstandardisasi bagi siapa pun yang mengakses repositori ini.

## Struktur Direktori

```
MLOps-DBD/
├── .devcontainer/
│   └── devcontainer.json      # Konfigurasi GitHub Codespaces (Python 3.11 + dependencies)
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml    # CI pipeline: ingest -> validate -> train (lihat bagian di bawah)
├── config/
│   └── config.yaml            # Parameter proyek, ingestion, validasi & eksperimen
├── data/
│   ├── raw/                   # Snapshot data mentah hasil ingestion (di-ignore git)
│   │   └── latest.csv         # Snapshot terbaru, acuan tahap validasi & training
│   ├── interim/                # Data hasil pembersihan awal (di-ignore git)
│   ├── processed/              # Data siap-latih / fitur final (di-ignore git)
│   ├── external/                # Data pihak ketiga (di-ignore git)
│   └── validation/             # Laporan JSON hasil validasi data (di-ignore git)
├── models/                    # Artefak model terlatih (.pkl, dll — di-ignore git)
├── notebooks/                 # Notebook eksplorasi & eksperimen
│   └── 0.1-initial-eda.ipynb  # Notebook EDA awal
├── src/                       # Source code modular (dapat diimpor sebagai package)
│   ├── ingestion/
│   │   └── fetch_data.py      # Tarik data terbaru dari API eksternal -> data/raw/
│   ├── data/
│   │   ├── make_dataset.py    # Ambil raw data -> bersihkan -> interim
│   │   └── validate_dataset.py # Gate validasi: data harus lolos sebelum training
│   ├── features/
│   │   └── build_features.py  # interim -> feature engineering -> processed
│   ├── models/
│   │   ├── train_model.py     # Melatih model dari data processed
│   │   └── predict_model.py   # Prediksi menggunakan model terlatih
│   └── visualization/
│       └── visualize.py       # Fungsi bantu untuk plotting
├── .gitignore
├── LICENSE                    # MIT License
├── requirements.txt           # Dependency Python
└── README.md
```

Struktur ini mengikuti konvensi **Cookiecutter Data Science**: data mentah tidak pernah ditimpa secara destruktif (`data/raw` menyimpan snapshot bertimestamp), setiap tahap pemrosesan data punya folder sendiri (`interim` → `processed`), dan logika yang dapat dipakai ulang dipisahkan ke `src/` alih-alih ditulis berulang di notebook.

## Data Ingestion & Trigger Otomatis (Pengembangan Lanjutan)

Di luar rubrik LK1, proyek ini juga mengimplementasikan alur MLOps yang lebih lengkap: **data tidak langsung memicu training begitu masuk** — ada gate validasi di antaranya. Alur ini dijalankan otomatis lewat `.github/workflows/ml-pipeline.yml`:

```
[schedule / manual trigger]
        │
        ▼
┌───────────────┐     data baru      ┌──────────────────┐   lolos?   ┌───────────────┐
│  1. Ingest     │ ──────────────────▶│  2. Validate      │ ─────────▶│  3. Train      │
│  fetch_data.py │   data/raw/latest  │  validate_dataset │  (gate)   │  train_model.py│
└───────────────┘                    └──────────────────┘            └───────────────┘
                                              │
                                        gagal │
                                              ▼
                                     job berhenti, training
                                     di-skip otomatis
```

1. **Ingest** — `src/ingestion/fetch_data.py` menarik data dari API eksternal (URL & auth diatur lewat `config/config.yaml` + secret `DBD_API_KEY`), lalu menyimpan snapshot bertimestamp ke `data/raw/` dan memperbarui `data/raw/latest.csv`.
2. **Validate (gate)** — `src/data/validate_dataset.py` memeriksa skema, missing value, jumlah baris minimum, dan duplikasi. Kalau ada yang tidak lolos, script exit dengan kode error dan pipeline **berhenti di sini** — training tidak dijalankan sama sekali.
3. **Train** — hanya berjalan kalau job validasi sukses (dependensi `needs: validate` di workflow). Menjalankan `build_features.py` lalu `train_model.py`, hasil model diunggah sebagai artifact.

**Trigger workflow:**
- `schedule` (cron) — berjalan otomatis berkala untuk menarik data terbaru.
- `workflow_dispatch` — bisa dipicu manual dari tab **Actions** di GitHub kapan pun dibutuhkan.

**Setup yang perlu dilakukan sebelum workflow ini aktif:**
1. Ganti `ingestion.api_url` di `config/config.yaml` dengan endpoint API data DBD yang sebenarnya, dan `validation.required_columns` dengan skema data yang sesuai.
2. Tambahkan secret `DBD_API_KEY` di **Settings → Secrets and variables → Actions** (kalau API-nya butuh auth).
3. Sesuaikan jadwal cron di `ml-pipeline.yml` sesuai kebutuhan (default: tiap hari jam 03:00 UTC).

## Cara Menjalankan Lingkungan via GitHub Codespaces

1. Buka repositori ini di GitHub, klik tombol **Code → Codespaces → Create codespace on main**.
2. Tunggu Codespaces membangun container sesuai `.devcontainer/devcontainer.json` (Python 3.11 + seluruh dependency di `requirements.txt` terinstal otomatis via `postCreateCommand`).
3. Setelah environment siap, verifikasi instalasi:
   ```bash
   python --version
   pip list
   ```
4. Jalankan notebook EDA awal lewat ekstensi Jupyter yang sudah ter-install, atau via terminal:
   ```bash
   jupyter notebook notebooks/0.1-initial-eda.ipynb
   ```
5. Untuk menjalankan pipeline dari terminal (setelah `data/raw/dataset.csv` tersedia):
   ```bash
   python -m src.data.make_dataset
   python -m src.features.build_features
   python -m src.models.train_model
   ```

### Menjalankan secara lokal (alternatif tanpa Codespaces)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Strategi Branching (GitHub Flow)

- `main` — branch stabil, hanya menerima merge dari branch fitur yang sudah divalidasi (lewat Pull Request).
- `feat/initial-eda` — branch kerja untuk eksperimen/eksplorasi awal (initial EDA). Branch fitur berikutnya mengikuti pola `feat/<deskripsi-singkat>`.

Alur kerja:
1. Buat branch baru dari `main`: `git checkout -b feat/nama-fitur`.
2. Commit perubahan secara bertahap dengan pesan yang informatif.
3. Push branch dan buka Pull Request ke `main`.
4. Setelah direview/divalidasi, merge PR ke `main`.

## Lisensi

Didistribusikan di bawah lisensi [MIT](LICENSE).
