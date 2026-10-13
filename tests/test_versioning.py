import pandas as pd

from src.data.versioning import diff_datasets, merge_into_dataset
from src.schema import DATE, FETCHED_AT, PAGE_VIEWS, PRECIPITATION, RAW_COLUMNS


def _fetch(dates, page_views, fetched_at):
    df = pd.DataFrame({DATE: pd.date_range(dates[0], dates[1], freq="D").date})
    df[PAGE_VIEWS] = page_views
    df[FETCHED_AT] = fetched_at
    return df.reindex(columns=RAW_COLUMNS).fillna({PRECIPITATION: 0.0})


def test_new_dates_are_appended_not_overwritten():
    v1 = merge_into_dataset(None, _fetch(("2026-09-01", "2026-09-30"), 10, "2026-10-01T00:00:00"))
    v2 = merge_into_dataset(v1, _fetch(("2026-09-27", "2026-10-09"), 10, "2026-10-11T00:00:00"))

    assert len(v1) == 30
    assert len(v2) == 39  # 30 lama + 9 hari baru (1-9 Okt), overlap 27-30 Sep tidak dobel
    assert not v2[DATE].duplicated().any()
    assert v2[DATE].is_monotonic_increasing


def test_overlapping_date_takes_latest_fetch():
    v1 = merge_into_dataset(None, _fetch(("2026-09-01", "2026-09-10"), 10, "2026-10-01T00:00:00"))
    v2 = merge_into_dataset(v1, _fetch(("2026-09-10", "2026-09-10"), 99, "2026-10-11T00:00:00"))

    assert v2.set_index(DATE).loc[pd.Timestamp("2026-09-10").date(), PAGE_VIEWS] == 99


def test_diff_reports_added_and_revised_rows():
    v1 = merge_into_dataset(None, _fetch(("2026-09-01", "2026-09-10"), 10, "2026-10-01T00:00:00"))
    v2 = merge_into_dataset(v1, _fetch(("2026-09-10", "2026-09-12"), 99, "2026-10-11T00:00:00"))

    diff = diff_datasets(v1, v2)
    assert diff["rows_added"] == 2
    assert diff["rows_changed"] == 1
    assert diff["changed_per_column"] == {PAGE_VIEWS: 1}
    assert diff["added_dates"] == ["2026-09-11", "2026-09-12"]
