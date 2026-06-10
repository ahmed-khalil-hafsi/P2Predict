import pytest

from p2predict.input_checks import check_csv_sanity


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


def test_check_csv_sanity_keeps_na_rows(csv_path_with_nas):
    # NAs are no longer dropped at load time — that decision moved downstream
    # (target-NA drop + per-family feature handling) so we don't silently
    # discard rows whose only NA is in an unselected column.
    df = check_csv_sanity(csv_path_with_nas)
    # The fixture has 4 rows, 2 with NA values. All 4 must survive load.
    assert len(df) == 4
    assert df.isna().sum().sum() > 0


def test_check_csv_sanity_na_warning_goes_to_stderr_not_stdout(
    csv_path_with_nas, capsys
):
    # Under `--json` the train CLI's stdout must stay pure JSON. The NA
    # warning is routed to stderr so it can't precede/corrupt the document.
    check_csv_sanity(csv_path_with_nas)
    captured = capsys.readouterr()
    assert "missing values" in captured.err
    assert captured.out == ""
