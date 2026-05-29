import pytest

from modules.input_checks import check_csv_sanity


def test_check_csv_sanity_loads_clean_csv(csv_path_clean):
    df = check_csv_sanity(csv_path_clean)
    assert not df.empty
    assert "Price" in df.columns


def test_check_csv_sanity_aborts_on_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        check_csv_sanity(str(tmp_path / "does_not_exist.csv"))


def test_check_csv_sanity_aborts_on_empty_file(csv_path_empty):
    with pytest.raises(SystemExit):
        check_csv_sanity(csv_path_empty)


def test_check_csv_sanity_drops_na_rows_instead_of_aborting(csv_path_with_nas):
    df = check_csv_sanity(csv_path_with_nas)
    # The fixture has 4 rows, 2 with NA values.
    assert not df.empty
    assert df.isna().sum().sum() == 0
    assert len(df) == 2
