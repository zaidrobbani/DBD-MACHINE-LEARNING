# MLOps-DBD

Fondasi teknis proyek MLOps untuk membangun model *machine learning* untuk memprediksi (forecasting) minat masyarakat terhadap **DBD (Demam Berdarah Dengue)** berbasis data time-series. Proyek ini dibangun agar **reproducible** (dapat dijalankan ulang di lingkungan mana pun melalui GitHub Codespaces) dan dikelola dengan strategi branching **GitHub Flow**.

## Tujuan Proyek

Membangun pipeline machine learning end-to-end (mulai dari eksplorasi data, feature engineering, training, hingga evaluasi model) untuk memprediksi minat masyarakat terhadap DBD seminggu ke depan berdasarkan data yang tersedia, dengan lingkungan kerja yang konsisten dan terstandardisasi bagi siapa pun yang mengakses repositori ini.

## Struktur Direktori

```
MLOps-DBD/
├── .devcontainer/
│   └── devcontainer.json      # Konfigurasi GitHub Codespaces (Python 3.11 + dependencies)
├── .github/
│   ├── actions/setup-env/     # Composite action: setup Python + install dependency (dipakai semua job)
│   ├── actions/setup-dvc/     # Composite action: kredensial remote DVC (DagsHub) + identitas git bot
│   └── workflows/
│       └── ml-pipeline.yml    # CI/CT: test -> ingest -> validate -> drift -> train (lihat bagian di bawah)
├── config/
│   └── config.yaml            # Parameter proyek, ingestion, validasi & eksperimen
├── data/
│   ├── raw/
│   │   ├── dataset.csv        # Dataset mentah kumulatif (isinya di DVC, bukan di git)
│   │   ├── dataset.csv.dvc    # Pointer DVC (hash md5 + ukuran) — di-commit ke git
│   │   └── .gitignore         # Dibuat otomatis oleh `dvc add`
│   ├── interim/                # Data hasil pembersihan awal (di-ignore git)
│   ├── processed/              # Data siap-latih / fitur final (di-ignore git)
│   ├── external/                # Data pihak ketiga (di-ignore git)
│   └── validation/             # Laporan JSON hasil validasi data (di-ignore git)
├── models/                    # Artefak model (isinya di DVC); hanya metrics.json yang di-commit ke git
├── tests/                     # Unit test (pytest) yang dijalankan di CI
├── notebooks/                 # Notebook eksplorasi & eksperimen
│   └── 0.1-initial-eda.ipynb  # Notebook EDA awal
├── src/                       # Source code modular (dapat diimpor sebagai package)
│   ├── schema.py              # Satu-satunya sumber nama kolom dataset
│   ├── utils.py               # Helper bersama: load_config, argparse --config, reindex_daily
│   ├── ingestion/
│   │   ├── ingestion_data.py  # Pengumpul data: Wikipedia Pageviews + Open-Meteo -> data/raw/
│   │   └── verify_data_access.py # Cek akses kedua API (memakai ulang fungsi ingestion, tanpa menyimpan file)
│   ├── data/
│   │   ├── versioning.py      # Logika dataset kumulatif: merge data baru + diff dua versi
│   │   ├── compare_versions.py # Bandingkan isi dua versi dataset DVC (per commit git)
│   │   ├── validate_dataset.py # Aturan validasi tunggal (run_checks) + gate data mentah
│   │   └── preprocess.py      # dataset mentah -> reindex harian, interpolasi, validasi -> interim
│   ├── features/
│   │   └── build_features.py  # interim -> fitur lag/rolling + target 7 hari ke depan -> processed
│   ├── models/
│   │   ├── train_model.py     # Training + validasi temporal + perbandingan dengan baseline
│   │   ├── residual.py        # Wrapper model yang memprediksi selisih dari rata-rata 7 hari
│   │   └── predict_model.py   # Forecast 7 hari ke depan dari data terbaru
│   ├── monitoring/
│   │   └── check_drift.py     # Putuskan retrain di luar jadwal: data drift, performance drift, umur model
│   └── visualization/
│       └── visualize.py       # Plot forecast vs aktual periode test (dipakai train_model)
├── dvc.yaml                   # Pipeline DVC: validate -> preprocess -> features -> train
├── dvc.lock                   # Hash input/output tiap stage (dibuat `dvc repro`, di-commit ke git)
├── .gitignore
├── LICENSE                    # MIT License
├── requirements.txt           # Dependency Python
└── README.md
```

Struktur ini mengikuti konvensi **Cookiecutter Data Science**: data mentah tidak pernah hilang (setiap versi `data/raw/dataset.csv` tersimpan di DVC), setiap tahap pemrosesan data punya folder sendiri (`interim` → `processed`), dan logika yang dapat dipakai ulang dipisahkan ke `src/` alih-alih ditulis berulang di notebook.

## Pengumpulan Data (Data Collection)

Data mentah dikumpulkan oleh `src/ingestion/ingestion_data.py` dari dua API publik yang **tidak membutuhkan API key**:

| Sumber | Data yang diambil | Kolom hasil |
|---|---|---|
| [Wikipedia Pageviews API](https://wikimedia.org/api/rest_v1/) | Jumlah kunjungan harian artikel [Demam berdarah dengue](https://id.wikipedia.org/wiki/Demam_berdarah_dengue) di Wikipedia Bahasa Indonesia, sebagai proksi minat/kepedulian masyarakat terhadap DBD | `page_views` |
| [Open-Meteo API](https://open-meteo.com/) (Archive API untuk histori, Forecast API untuk beberapa hari terakhir) | Cuaca harian di Malang, Jawa Timur: curah hujan, suhu rata-rata, kelembapan rata-rata | `precipitation_sum`, `temperature_mean`, `relative_humidity_mean` |

Kedua sumber digabung berdasarkan kolom `date`, lalu ditambah kolom `fetched_at` (waktu pengambilan data).

### Menjalankan skrip pengumpul data

Jalankan dari **root repositori** (path config bersifat relatif) dengan environment yang sudah aktif:

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 1. Ambil data terbaru dan tambahkan ke data/raw/dataset.csv
python -m src.ingestion.ingestion_data

# 2. (Opsional) Bersihkan & validasi data mentah -> data/interim/
python -m src.data.preprocess
```

Opsi yang tersedia:

```bash
# Ambil histori panjang untuk training (ingestion.history_days, default 730 hari)
python -m src.ingestion.ingestion_data --backfill

# Ambil rentang tertentu (default: 14 hari, dari ingestion.days_back)
python -m src.ingestion.ingestion_data --days-back 60

# Tentukan tanggal akhir sendiri (default: hari ini - ingestion.end_date_offset_days)
python -m src.ingestion.ingestion_data --backfill --end-date 2026-09-30

# Pakai file config lain
python -m src.ingestion.ingestion_data --config path/ke/config.yaml
python -m src.data.preprocess --config path/ke/config.yaml
```

Contoh output:

```
Mengambil data untuk rentang 2026-09-26 sampai 2026-10-09
Wikipedia Pageviews API: 14 baris berhasil diambil
Open-Meteo API: 14 baris berhasil diambil
Dataset diperbarui di data/raw/dataset.csv: 730 -> 739 baris (+9 baru, 0 direvisi)
Rentang data: 2024-10-01 s/d 2026-10-09
```

### Hasil yang dihasilkan

| Skrip | File output | Keterangan |
|---|---|---|
| `ingestion_data.py` | `data/raw/dataset.csv` | Dataset kumulatif: data baru **ditambahkan**, tanggal yang sudah ada diganti hasil fetch terbaru (satu baris per tanggal). Riwayat versinya disimpan DVC |
| `preprocess.py` | `data/interim/dataset_clean.csv` | Data bersih: satu baris per hari kalender, interpolasi linear kolom cuaca untuk celah ≤ `preprocessing.max_interpolate_gap_days` hari |

`preprocess.py` akan exit dengan kode error jika data bersih tidak lolos validasi. Aturannya sama dengan gate data mentah (`run_checks` di `validate_dataset.py`, parameter di `validation`), jadi tidak ada dua versi logika validasi.

### Konfigurasi

Semua parameter ada di `config/config.yaml`:

- `location` — koordinat lokasi data cuaca (default: Malang).
- `wikipedia` — artikel, bahasa, dan `user_agent` untuk request ke Wikimedia (wajib diisi kontak yang valid sesuai kebijakan Wikimedia).
- `paths` — semua lokasi file input/output pipeline (satu tempat).
- `open_meteo` — URL Archive & Forecast API. Variabel cuaca yang diambil didefinisikan di `src/schema.py`.
- `ingestion.days_back` — jumlah hari yang diambil; `ingestion.history_days` — panjang histori untuk `--backfill`; `ingestion.end_date_offset_days` — jeda hari dari hari ini (default 2 hari, karena data pageviews baru tersedia setelah ~1 hari).
- `preprocessing.max_interpolate_gap_days` — celah data cuaca terpanjang yang boleh diinterpolasi.
- `validation` — ambang batas validasi (dipakai untuk data mentah dan data bersih).
- `features`, `split`, `model` — parameter fitur, validasi temporal, dan model (`random_state` diambil dari `project.seed`).

### Versioning data dengan DVC

Git hanya menyimpan **pointer** kecil `data/raw/dataset.csv.dvc` (hash md5 + ukuran). Isi CSV-nya disimpan di cache DVC (`.dvc/cache`) dan remote DVC. Setiap commit git yang mengubah pointer itu adalah satu versi dataset.

**Kenapa dataset kumulatif (bukan snapshot yang ditimpa):** ingestion harian hanya mengambil jendela 14 hari terakhir. Kalau hasilnya menimpa file, datanya bergeser, bukan bertambah, dan model kehilangan histori. Karena itu `ingestion_data.py` menggabungkan data baru ke `dataset.csv` (`merge_into_dataset` di `src/data/versioning.py`), lalu DVC yang menyimpan setiap versinya. Snapshot bertimestamp tidak dibutuhkan lagi karena DVC sudah menyimpan riwayatnya.

Alur menambah versi data baru:

```bash
python -m src.ingestion.ingestion_data          # 1. tambah data baru ke dataset.csv
dvc status                                      # 2. DVC mendeteksi dataset.csv berubah
python -m src.data.compare_versions             # 3. cek isi perubahan (HEAD vs workspace)
dvc add data/raw/dataset.csv                    # 4. hash baru ditulis ke dataset.csv.dvc
git add data/raw/dataset.csv.dvc
git commit -m "data: tambah data s/d <tanggal>"
dvc push                                        # 5. kirim isi data ke remote DVC
```

Audit riwayat versi:

```bash
git log --oneline -- data/raw/dataset.csv.dvc               # daftar versi dataset
dvc diff --show-hash HEAD~1 HEAD                            # hash yang berubah antar versi
python -m src.data.compare_versions --old HEAD~1 --new HEAD # baris yang bertambah / berubah
```

`dvc diff` hanya menunjukkan **file mana** yang berubah beserta hash-nya. `compare_versions` membuka kedua versi dari cache DVC (`dvc.api`) dan menunjukkan **isi** perubahannya: jumlah baris, rentang tanggal, baris baru, dan baris yang direvisi. md5 yang dicetaknya sama dengan md5 di file `.dvc`.

Kembali ke versi lama (mis. untuk reproduksi training):

```bash
git checkout <commit> -- data/raw/dataset.csv.dvc
dvc checkout
```

> `data/raw/` sengaja **tidak** di-ignore di `.gitignore` root. Kalau di-ignore, DVC ikut melewatkan file `.dvc` di folder itu (`dvc status` menjawab "no data tracked"). Yang di-ignore cukup `dataset.csv`, lewat `data/raw/.gitignore` yang ditulis `dvc add`.

### Pipeline DVC (`dvc.yaml`)

Tahap `validate → preprocess → features → train` didefinisikan di `dvc.yaml`, dengan parameter dibaca langsung dari `config/config.yaml`. `dvc repro` hanya menjalankan ulang tahap yang data, kode, atau parameternya berubah. Ingestion tidak jadi stage karena hasilnya bergantung pada API eksternal.

```bash
dvc repro                 # jalankan tahap yang berubah saja
dvc repro -f train        # paksa train ulang
dvc dag                   # alur antar tahap
dvc metrics show          # metrik model (models/metrics.json, di-commit ke git)
dvc metrics diff main     # bandingkan metrik dengan branch main
dvc params diff           # parameter config yang berubah
```

Setelah `dvc repro`, commit `dvc.lock` dan `models/metrics.json`, lalu `dvc push` (model dan data antara disimpan di remote DVC).

### Retrain berbasis drift (`src/monitoring/check_drift.py`)

Selain retrain terjadwal, model dilatih ulang kalau salah satu sinyal ini terpenuhi (ambang di `monitoring` pada config):

| Sinyal | Kondisi |
|---|---|
| Data drift | Rata-rata `page_views` 7 hari terakhir punya \|z-score\| > 3 terhadap rata-rata mingguan 90 hari sebelumnya |
| Performance drift | MAE pada data yang baru berlabel (sesudah `trained_until`) > 1.5× MAE test |
| Umur model | Model lebih tua dari 10 hari (pengaman kalau retrain terjadwal gagal) |
| Model belum ada | Selalu retrain |

Sinyal drift diabaikan selama 3 hari setelah training (`drift_cooldown_days`), supaya lonjakan yang bertahan lama tidak memicu training setiap hari.

```bash
python -m src.monitoring.check_drift    # cetak retrain=true/false beserta alasannya
```

Di GitHub Actions, keputusan ditulis ke `$GITHUB_OUTPUT` (`retrain=true|false`) dan rinciannya ke job summary.

## Pipeline ML & CI/CT

### Formulasi masalah

Data kasus DBD resmi tidak tersedia lewat API publik, jadi proyek ini memakai **page_views artikel DBD di Wikipedia** sebagai proksi minat/kepedulian masyarakat, ditambah cuaca Malang sebagai fitur.

- **Task**: regresi time-series (forecasting).
- **Target**: rata-rata `page_views` pada **t+1 .. t+7** (minggu depan).
- **Fitur** (hanya informasi sampai hari t): kalender (hari, bulan, akhir pekan), lag `page_views` (1, 2, 3, 7, 14 hari), rata-rata bergerak 7/14/28 hari, tren, serta agregat cuaca 7/14/28 hari dan versi yang digeser 7/14 hari (efek cuaca ke populasi nyamuk muncul dengan jeda 1-3 minggu).
- **Model**: Random Forest yang memprediksi *selisih* terhadap rata-rata 7 hari terakhir (`model.baseline_feature` + `model.residual`), karena level page_views bergeser antar tahun dan model berbasis pohon tidak bisa memprediksi di luar rentang nilai saat training.

### Validasi temporal

Data time-series **tidak boleh di-split acak** (`train_test_split` dengan shuffle), karena model akan "melihat masa depan".

1. **Holdout kronologis**: 90 hari terakhir (`split.test_days`) jadi test set.
2. **Purge gap**: baris train yang target-nya (t+1..t+7) masuk periode test dibuang, jadi ada jeda 7 hari antara train dan test.
3. **CV temporal**: `TimeSeriesSplit` (expanding window, 5 fold, `gap=7`) pada data train untuk memilih hyperparameter. Test set tidak dipakai untuk tuning.
4. **Baseline naif**: "minggu depan = rata-rata 7 hari terakhir". Metrik model (MAE, RMSE, R²) selalu dibandingkan dengan baseline ini dan disimpan di `models/metrics.json`, beserta grafik `models/forecast_test.png`.
5. Setelah dievaluasi, model dilatih ulang dengan seluruh data berlabel (`model.refit_on_full_data`) sebelum disimpan.

### Alur CI/CT (`.github/workflows/ml-pipeline.yml`)

```
push / PR ──────────▶ 1. Test (CI): ruff + pytest ── selesai

cron / manual ──────▶ 1. Test ─▶ 2. Data ─────────────────────────────▶ 3. Train (CT)
                                 dvc pull                               (hanya jika perlu)
                                 ingest incremental (14 hari)           dvc pull
                                 validate (gate)                        dvc repro --force
                                 cek drift ─▶ retrain=true/false        predict
                                 dvc add + dvc push + git commit        dvc push + git commit
```

| Trigger | Jadwal (WIB) | Train jalan kalau |
|---|---|---|
| Push / PR | — | tidak pernah (test saja) |
| Cek harian | Selasa–Minggu 10:17 | `check_drift` memutuskan `retrain=true` |
| Retrain mingguan | Senin 10:17 | selalu |
| Manual (**Actions → Run workflow**) | kapan saja | `force_train` dicentang, atau drift |

1. **Test**: `ruff check` + `pytest` (unit test anti-leakage fitur, split temporal, aturan validasi, cek drift). Tanpa akses jaringan.
2. **Data**: `dvc pull` dataset kumulatif dan model terakhir dari DagsHub, tambah data 14 hari terakhir, lalu **validasi (gate)**: cek kolom wajib, jumlah baris, tanggal duplikat, missing value, rentang nilai, dan kesegaran data. Gagal = tidak ada commit dan training tidak jalan. Setelah itu `check_drift` memutuskan perlu retrain atau tidak, dan versi dataset baru di-`dvc push` + di-commit (`data: ingest s/d <tanggal>`).
3. **Train**: `dvc repro --force` (validate → preprocess → features → train) → forecast 7 hari. Metrik dan alasan retrain tampil di *job summary*. Model di-`dvc push`, `dvc.lock` + `metrics.json` di-commit (`model: retrain (<alasan>)`), dan model juga diunggah sebagai artifact.

Kebutuhan:
- Secret repo **`DAGSHUB_TOKEN`** (Settings → Secrets and variables → Actions) berisi token DagsHub. Kredensial ditulis ke `.dvc/config.local` oleh `.github/actions/setup-dvc`, tidak pernah di-commit.
- Jadwal cron hanya dijalankan GitHub dari workflow di branch **`main`**.
- Pipeline meng-commit ke branch yang dijalankan, jadi jalankan `git pull` sebelum lanjut bekerja di lokal.

Jalankan pipeline yang sama secara lokal:

```bash
dvc pull                                     # dataset + model terakhir dari DagsHub
python -m src.ingestion.ingestion_data       # tambah data terbaru
python -m src.monitoring.check_drift         # perlu retrain?
dvc repro                                    # validate -> preprocess -> features -> train
python -m src.models.predict_model
python -m pytest -q && ruff check src tests
```

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
5. Untuk menjalankan pipeline dari terminal, ikuti perintah di bagian **Pipeline ML & CI/CT**.

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
