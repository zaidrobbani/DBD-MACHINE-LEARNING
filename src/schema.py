"""Satu-satunya sumber nama kolom dataset. Modul lain wajib import dari sini."""

DATE = "date"
PAGE_VIEWS = "page_views"
FETCHED_AT = "fetched_at"
TARGET = "target"
PRECIPITATION = "precipitation_sum"
TEMPERATURE = "temperature_mean"
HUMIDITY = "relative_humidity_mean"

# Nama variabel Open-Meteo -> nama kolom di dataset.
OPEN_METEO_VARIABLES = {
    "precipitation_sum": PRECIPITATION,
    "temperature_2m_mean": TEMPERATURE,
    "relative_humidity_2m_mean": HUMIDITY,
}
WEATHER_COLUMNS = list(OPEN_METEO_VARIABLES.values())

VALUE_COLUMNS = [PAGE_VIEWS, *WEATHER_COLUMNS]
REQUIRED_COLUMNS = [DATE, *VALUE_COLUMNS]
RAW_COLUMNS = [*REQUIRED_COLUMNS, FETCHED_AT]
