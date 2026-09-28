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
│   │   ├── ingestion_data.py  # Pengumpul data: Wikipedia Pageviews + Open-Meteo -> data/raw/
│   │   └── fetch_data.py      # Tarik data terbaru dari API eksternal -> data/raw/
│   ├── data/
│   │   ├── preprocess.py      # Gabung snapshot raw -> dedup, interpolasi, validasi -> interim
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

## Pengumpulan Data (Data Collection)

Data mentah dikumpulkan oleh `src/ingestion/ingestion_data.py` dari dua API publik yang **tidak membutuhkan API key**:

| Sumber | Data yang diambil | Kolom hasil |
|---|---|---|
| [Wikipedia Pageviews API](https://wikimedia.org/api/rest_v1/) | Jumlah kunjungan harian artikel [Demam berdarah dengue](https://id.wikipedia.org/wiki/Demam_berdarah_dengue) di Wikipedia Bahasa Indonesia, sebagai proksi minat/kepedulian masyarakat terhadap DBD | `page_views` |
| [Open-Meteo API](https://open-meteo.com/) | Cuaca harian di Malang, Jawa Timur: curah hujan, suhu rata-rata, kelembapan rata-rata | `precipitation_sum`, `temperature_mean`, `relative_humidity_mean` |

Kedua sumber digabung berdasarkan kolom `date`, lalu ditambah kolom `fetched_at` (waktu pengambilan data).

### Menjalankan skrip pengumpul data

Jalankan dari **root repositori** (path config bersifat relatif) dengan environment yang sudah aktif:

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 1. Ambil data mentah -> data/raw/
python src/ingestion/ingestion_data.py

# 2. (Opsional) Bersihkan & validasi data mentah -> data/interim/
python src/data/preprocess.py
```

Opsi yang tersedia:

```bash
# Ambil rentang yang lebih panjang (default: 14 hari, dari ingestion.days_back)
python src/ingestion/ingestion_data.py --days-back 60

# Pakai file config lain
python src/ingestion/ingestion_data.py --config path/ke/config.yaml
python src/data/preprocess.py --config path/ke/config.yaml
```

Contoh output:

```
Mengambil data untuk rentang 2026-09-13 sampai 2026-09-26
Wikipedia Pageviews API: 14 baris berhasil diambil
Open-Meteo API: 14 baris berhasil diambil
Snapshot baru disimpan di data/raw/raw_20260928_141123.csv
Pointer terbaru diperbarui di data/raw/latest.csv
```

### Hasil yang dihasilkan

| Skrip | File output | Keterangan |
|---|---|---|
| `ingestion_data.py` | `data/raw/raw_YYYYMMDD_HHMMSS.csv` | Snapshot bertimestamp, dibuat baru setiap kali dijalankan (tidak menimpa snapshot lama) |
| | `data/raw/latest.csv` | Salinan snapshot terakhir |
| `preprocess.py` | `data/interim/clean_YYYYMMDD_HHMMSS.csv` | Data bersih: gabungan semua snapshot, deduplikasi per tanggal (ambil `fetched_at` terbaru), interpolasi linear kolom cuaca, baris tanpa `page_views` dibuang |
| | `data/interim/latest.csv` | Salinan data bersih terakhir |

`preprocess.py` akan exit dengan kode error jika data tidak lolos validasi: kolom wajib hilang, jumlah baris di bawah `preprocessing.min_rows`, masih ada tanggal duplikat, atau rasio missing value melebihi `preprocessing.missing_value_threshold`.

### Konfigurasi

Semua parameter ada di `config/config.yaml`:

- `location` — koordinat lokasi data cuaca (default: Malang).
- `wikipedia` — artikel, bahasa, dan `user_agent` untuk request ke Wikimedia (wajib diisi kontak yang valid sesuai kebijakan Wikimedia).
- `open_meteo.daily_variables` — variabel cuaca harian yang diambil.
- `ingestion.days_back` — jumlah hari yang diambil; `ingestion.end_date_offset_days` — jeda hari dari hari ini (default 2 hari, karena data pageviews baru tersedia setelah ~1 hari).
- `preprocessing` — folder input/output dan ambang batas validasi.

### Versioning data dengan DVC

`data/raw/*` di-ignore oleh git, sedangkan `data/raw/latest.csv` di-track oleh DVC. Setelah menjalankan ingestion, perbarui pointer DVC-nya:

```bash
dvc add data/raw/latest.csv
git add data/raw/latest.csv.dvc
dvc push   # jika remote DVC sudah dikonfigurasi
```

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
