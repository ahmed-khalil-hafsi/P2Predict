from modules.prepare_data import prepare_data


def test_prepare_data_random_split_excludes_target(synthetic_parts):
    X_train, X_test, y_train, y_test, _, _ = prepare_data(
        synthetic_parts, ["Weight", "Region", "Supplier", "Size"], "Price"
    )
    assert "Price" not in X_train.columns
    assert len(X_train) + len(X_test) == len(synthetic_parts)


def test_prepare_data_time_aware_split_is_chronological(synthetic_parts_with_date):
    X_train, X_test, _, _, _, _ = prepare_data(
        synthetic_parts_with_date,
        ["Weight", "Region", "Supplier", "Size", "Date"],
        "Price",
        time_column="Date",
    )
    # Date is excluded from the feature matrix.
    assert "Date" not in X_train.columns
    # And the test set lives entirely after the train set in row order.
    assert X_test.index.min() > X_train.index.max()


def test_prepare_data_time_aware_drops_time_from_selected(synthetic_parts_with_date):
    # Caller passes Date in selected_columns; prepare_data must strip it.
    X_train, _, _, _, num, cat = prepare_data(
        synthetic_parts_with_date,
        ["Weight", "Date"],
        "Price",
        time_column="Date",
    )
    assert list(X_train.columns) == ["Weight"]
    assert "Date" not in list(num) + list(cat)
