import pandas as pd
import numpy as np
import pathlib
import sys

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))

import tablemage as tm


def test_bool_columns_converted_to_int():
    """Tests that native bool dtype columns are converted to int on init."""
    df = pd.DataFrame(
        {
            "bool_col": [True, False, True, True, False, False],
            "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    df_train = df.iloc[:4].copy()
    df_test = df.iloc[4:].copy()

    assert df_train["bool_col"].dtype == bool

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)

    result_train = analyzer.df_train()
    result_test = analyzer.df_test()

    assert result_train["bool_col"].dtype in (np.int64, np.int32, int)
    assert result_test["bool_col"].dtype in (np.int64, np.int32, int)
    assert list(result_train["bool_col"]) == [1, 0, 1, 1]
    assert list(result_test["bool_col"]) == [0, 0]


def test_bool_columns_classified_as_numeric():
    """Tests that bool columns (converted to int) are treated as numeric vars."""
    df = pd.DataFrame(
        {
            "bool_col": [True, False, True, True, False, False],
            "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    analyzer = tm.Analyzer(df, verbose=False)
    assert "bool_col" in analyzer.numeric_vars()


def test_string_true_false_normalized_to_lowercase():
    """Tests that string 'True'/'False' values are lowercased on init."""
    df = pd.DataFrame(
        {
            "str_bool_col": ["True", "False", "True", "False", "True", "False"],
            "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    df_train = df.iloc[:4].copy()
    df_test = df.iloc[4:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)

    result_train = analyzer.df_train()
    result_test = analyzer.df_test()

    assert all(v in ("true", "false") for v in result_train["str_bool_col"])
    assert all(v in ("true", "false") for v in result_test["str_bool_col"])


def test_mixed_case_true_false_normalized():
    """Tests that mixed case variations like 'TRUE', 'FALSE', 'True' are all
    lowercased to 'true'/'false'."""
    df = pd.DataFrame(
        {
            "str_bool_col": ["TRUE", "FALSE", "True", "False", "true", "false"],
            "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    df_train = df.iloc[:4].copy()
    df_test = df.iloc[4:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)

    result_train = analyzer.df_train()
    result_test = analyzer.df_test()

    assert list(result_train["str_bool_col"]) == ["true", "false", "true", "false"]
    assert list(result_test["str_bool_col"]) == ["true", "false"]


def test_non_boolean_strings_unaffected():
    """Tests that non-boolean string values in object columns are not modified."""
    df = pd.DataFrame(
        {
            "mixed_col": ["True", "cat", "False", "dog", "True", "cat"],
            "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    df_train = df.iloc[:4].copy()
    df_test = df.iloc[4:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)

    result_train = analyzer.df_train()
    result_test = analyzer.df_test()

    assert list(result_train["mixed_col"]) == ["true", "cat", "false", "dog"]
    assert list(result_test["mixed_col"]) == ["true", "cat"]


def test_bool_with_nan():
    """Tests that boolean-like columns with NaN values are handled properly."""
    df = pd.DataFrame(
        {
            "str_bool_col": ["True", None, "False", "True", None, "False"],
            "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    df_train = df.iloc[:4].copy()
    df_test = df.iloc[4:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)

    result_train = analyzer.df_train()
    result_test = analyzer.df_test()

    non_null_train = result_train["str_bool_col"].dropna().tolist()
    non_null_test = result_test["str_bool_col"].dropna().tolist()

    assert all(v in ("true", "false") for v in non_null_train)
    assert all(v in ("true", "false") for v in non_null_test)
    assert result_train["str_bool_col"].isna().sum() == 1
    assert result_test["str_bool_col"].isna().sum() == 1


def test_original_dataframe_not_mutated():
    """Tests that the original DataFrame is not modified when booleans are converted."""
    df = pd.DataFrame(
        {
            "bool_col": [True, False, True, False],
            "str_bool_col": ["True", "FALSE", "true", "False"],
        }
    )
    original_bool_values = df["bool_col"].tolist()
    original_str_values = df["str_bool_col"].tolist()

    tm.Analyzer(df, verbose=False)

    assert df["bool_col"].tolist() == original_bool_values
    assert df["str_bool_col"].tolist() == original_str_values


def test_no_bool_columns_no_side_effects():
    """Tests that DataFrames without any boolean values are unaffected."""
    df = pd.DataFrame(
        {
            "cat_col": ["a", "b", "c", "d", "e", "f"],
            "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    df_train = df.iloc[:4].copy()
    df_test = df.iloc[4:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)

    result_train = analyzer.df_train()
    assert list(result_train["cat_col"]) == ["a", "b", "c", "d"]


def test_emitter_bool_columns_converted_to_int():
    """Tests that bool columns are converted to int when emitted via DataEmitter."""
    df = pd.DataFrame(
        {
            "bool_col": [True, False, True, True, False, False, True, False],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    df_train = df.iloc[:6].copy()
    df_test = df.iloc[6:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)
    emitter = analyzer.datahandler().train_test_emitter(
        y_var="target", X_vars=["bool_col"]
    )
    X_train, _, X_test, _ = emitter.emit_train_test_Xy(verbose=False)

    assert X_train["bool_col"].dtype in (np.int64, np.int32, int, np.float64)
    assert X_test["bool_col"].dtype in (np.int64, np.int32, int, np.float64)
    assert set(X_train["bool_col"].unique()).issubset({0, 1, 0.0, 1.0})
    assert set(X_test["bool_col"].unique()).issubset({0, 1, 0.0, 1.0})


def test_emitter_string_true_false_consistent():
    """Tests that string 'True'/'False' values remain consistently lowercase
    when emitted via DataEmitter, avoiding mixed 'True'/'true' categories."""
    df = pd.DataFrame(
        {
            "str_bool_col": [
                "True",
                "False",
                "True",
                "False",
                "True",
                "False",
                "True",
                "False",
            ],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    df_train = df.iloc[:6].copy()
    df_test = df.iloc[6:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)
    emitter = analyzer.datahandler().train_test_emitter(
        y_var="target", X_vars=["str_bool_col"]
    )
    X_train, _, X_test, _ = emitter.emit_train_test_Xy(verbose=False)

    # after one-hot encoding, column names should use lowercase "true"/"false"
    onehot_cols = [c for c in X_train.columns if "str_bool_col" in c]
    for col in onehot_cols:
        assert (
            "True" not in col
        ), f"Column name '{col}' contains capitalized 'True'; expected lowercase"
        assert (
            "False" not in col
        ), f"Column name '{col}' contains capitalized 'False'; expected lowercase"


def test_emitter_mixed_case_true_false_consistent():
    """Tests that mixed case 'TRUE'/'FALSE'/'True'/'False' all produce consistent
    lowercase categories when emitted via DataEmitter."""
    df = pd.DataFrame(
        {
            "str_bool_col": [
                "TRUE",
                "FALSE",
                "True",
                "False",
                "true",
                "false",
                "TRUE",
                "FALSE",
            ],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    df_train = df.iloc[:6].copy()
    df_test = df.iloc[6:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)
    emitter = analyzer.datahandler().train_test_emitter(
        y_var="target", X_vars=["str_bool_col"]
    )
    X_train, _, X_test, _ = emitter.emit_train_test_Xy(verbose=False)

    # should only have 2 unique categories (true/false), not 4+ from mixed casing
    onehot_cols = [c for c in X_train.columns if "str_bool_col" in c]
    # with dropfirst=True and 2 categories, we expect exactly 1 one-hot column
    assert len(onehot_cols) == 1, (
        f"Expected 1 one-hot column (2 categories, drop first), "
        f"got {len(onehot_cols)}: {onehot_cols}"
    )


def test_emitter_string_true_false_as_target():
    """Tests that string 'True'/'False' target values are consistently lowercase
    when emitted via DataEmitter."""
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "target": [
                "True",
                "False",
                "TRUE",
                "FALSE",
                "True",
                "False",
                "true",
                "false",
            ],
        }
    )
    df_train = df.iloc[:6].copy()
    df_test = df.iloc[6:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)
    emitter = analyzer.datahandler().train_test_emitter(
        y_var="target", X_vars=["feature"]
    )
    _, y_train, _, y_test = emitter.emit_train_test_Xy(verbose=False)

    assert set(y_train.unique()).issubset(
        {"true", "false"}
    ), f"Target values contain non-lowercase booleans: {y_train.unique()}"
    assert set(y_test.unique()).issubset(
        {"true", "false"}
    ), f"Target values contain non-lowercase booleans: {y_test.unique()}"


def test_emitter_force_categorical_on_bool_int_column():
    """Tests that when a bool column (converted to int 0/1) is forced to categorical,
    the DataEmitter produces consistent string values without mixed casing."""
    df = pd.DataFrame(
        {
            "bool_col": [True, False, True, False, True, False, True, False],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    df_train = df.iloc[:6].copy()
    df_test = df.iloc[6:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)
    # bool was converted to int (0/1) on init; now force it to categorical
    analyzer.force_categorical(["bool_col"])

    emitter = analyzer.datahandler().train_test_emitter(
        y_var="target", X_vars=["bool_col"]
    )
    X_train, _, X_test, _ = emitter.emit_train_test_Xy(verbose=False)

    # should have consistent category names (e.g. "0"/"1"), not "True"/"False"
    onehot_cols = [c for c in X_train.columns if "bool_col" in c]
    for col in onehot_cols:
        assert "True" not in col, (
            f"Column name '{col}' contains 'True'; bool should have been int before "
            f"force_categorical, producing '0'/'1' categories"
        )
        assert "False" not in col, (
            f"Column name '{col}' contains 'False'; bool should have been int before "
            f"force_categorical, producing '0'/'1' categories"
        )


def test_emitter_categories_no_mixed_casing():
    """Tests that the DataEmitter's categorical variable categories don't contain
    both 'True' and 'true' (the exact bug that caused plotting issues)."""
    df = pd.DataFrame(
        {
            "str_bool_col": [
                "True",
                "False",
                "True",
                "False",
                "True",
                "False",
                "True",
                "False",
            ],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    df_train = df.iloc[:6].copy()
    df_test = df.iloc[6:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)

    # check that the working DataFrames in the DataHandler have consistent values
    handler_train = analyzer.datahandler().df_train()
    handler_test = analyzer.datahandler().df_test()

    train_unique = set(handler_train["str_bool_col"].dropna().unique())
    test_unique = set(handler_test["str_bool_col"].dropna().unique())

    # there should be no capitalized "True" or "False"
    assert (
        "True" not in train_unique
    ), f"Found 'True' in train categories: {train_unique}"
    assert (
        "False" not in train_unique
    ), f"Found 'False' in train categories: {train_unique}"
    assert "True" not in test_unique, f"Found 'True' in test categories: {test_unique}"
    assert (
        "False" not in test_unique
    ), f"Found 'False' in test categories: {test_unique}"

    # all values should be lowercase
    assert train_unique == {"true", "false"}
    assert test_unique == {"true", "false"}


def test_emitter_custom_transform_bool_handling():
    """Tests that custom_transform properly handles bool columns in new data."""
    df = pd.DataFrame(
        {
            "bool_col": [True, False, True, False, True, False, True, False],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    df_train = df.iloc[:6].copy()
    df_test = df.iloc[6:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)
    emitter = analyzer.datahandler().train_test_emitter(
        y_var="target", X_vars=["bool_col"]
    )
    # fit the emitter
    emitter.emit_train_test_Xy(verbose=False)

    # now transform new data with bool column
    new_data = pd.DataFrame({"bool_col": [True, False]})
    transformed = emitter.custom_transform(new_data)

    # bool should have been converted to int, not left as True/False strings
    assert transformed["bool_col"].dtype in (np.int64, np.int32, int, np.float64)
    assert set(transformed["bool_col"].unique()).issubset({0, 1, 0.0, 1.0})


def test_native_bool_in_object_column_normalized():
    """Tests that native Python bool values in an object column (common when
    pd.read_csv parses 'TRUE'/'FALSE') are normalized to lowercase strings."""
    # simulate what pd.read_csv produces: bool values + NaN in an object column
    df = pd.DataFrame(
        {
            "bool_obj_col": [True, False, True, None, False, True, False, True],
            "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    assert df["bool_obj_col"].dtype == object

    df_train = df.iloc[:6].copy()
    df_test = df.iloc[6:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)

    train = analyzer.df_train()
    test = analyzer.df_test()

    train_vals = train["bool_obj_col"].dropna().unique().tolist()
    test_vals = test["bool_obj_col"].dropna().unique().tolist()

    # all values should be lowercase strings, not Python bools
    assert all(
        isinstance(v, str) for v in train_vals
    ), f"Train contains non-string values: {train_vals}"
    assert all(
        isinstance(v, str) for v in test_vals
    ), f"Test contains non-string values: {test_vals}"
    assert set(train_vals).issubset({"true", "false"})
    assert set(test_vals).issubset({"true", "false"})


def test_native_bool_in_object_column_train_test_consistent():
    """Tests that train and test DataFrames have consistent types/values for
    native bool values in object columns — the root cause of the 4-bar plot bug."""
    df = pd.DataFrame(
        {
            "bool_obj_col": [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
            ],
            "numeric_col": list(range(10)),
        }
    )
    # force object dtype (simulating NaN presence in original data)
    df["bool_obj_col"] = df["bool_obj_col"].astype(object)

    analyzer = tm.Analyzer(df, test_size=0.3, verbose=False)

    train = analyzer.df_train()
    test = analyzer.df_test()

    train_unique = set(train["bool_obj_col"].dropna().unique())
    test_unique = set(test["bool_obj_col"].dropna().unique())

    # Both must be the same type and same values — no mixing of bool and str
    combined = train_unique | test_unique
    assert combined.issubset(
        {"true", "false"}
    ), f"Mixed types in train/test: train={train_unique}, test={test_unique}"


def test_df_all_no_mixed_bool_string_categories():
    """Tests that df_all() (train + test concatenated) has no mixed
    bool/string True/False categories — the exact bug from the EDA plot."""
    df = pd.DataFrame(
        {
            "bool_obj_col": [
                True,
                False,
                None,
                True,
                False,
                True,
                False,
                True,
                None,
                False,
            ],
            "numeric_col": list(range(10)),
        }
    )
    assert df["bool_obj_col"].dtype == object

    analyzer = tm.Analyzer(df, test_size=0.3, verbose=False)

    all_df = analyzer.datahandler().df_all()
    unique_vals = all_df["bool_obj_col"].dropna().unique().tolist()

    # must have exactly 2 categories, not 4
    assert (
        len(set(unique_vals)) == 2
    ), f"Expected 2 unique categories, got {len(set(unique_vals))}: {unique_vals}"
    assert set(unique_vals) == {
        "true",
        "false",
    }, f"Expected {{'true', 'false'}}, got {set(unique_vals)}"


def test_eda_categorical_vars_consistent_with_native_bools():
    """Tests that EDA report sees consistent categories when the input has
    native Python bools in object columns."""
    df = pd.DataFrame(
        {
            "bool_obj_col": [
                True,
                False,
                True,
                None,
                False,
                True,
                False,
                True,
                True,
                False,
            ],
            "numeric_col": list(range(10)),
        }
    )
    assert df["bool_obj_col"].dtype == object

    analyzer = tm.Analyzer(df, test_size=0.2, verbose=False)
    eda = analyzer.eda()

    # the EDA report should list this as a categorical var
    assert "bool_obj_col" in eda.categorical_vars()

    # check that the underlying DataFrame in the EDA has consistent categories
    eda_df = eda._df
    unique_vals = eda_df["bool_obj_col"].dropna().unique().tolist()
    assert set(unique_vals) == {
        "true",
        "false",
    }, f"EDA has inconsistent categories: {unique_vals}"


def test_emitter_native_bool_in_object_col_consistent():
    """Tests that the DataEmitter produces consistent one-hot encodings when
    the input has native Python bools in an object column."""
    df = pd.DataFrame(
        {
            "bool_obj_col": [
                True,
                False,
                True,
                None,
                False,
                True,
                False,
                True,
                True,
                False,
            ],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    assert df["bool_obj_col"].dtype == object

    df_train = df.iloc[:8].copy()
    df_test = df.iloc[8:].copy()

    analyzer = tm.Analyzer(df_train, df_test, verbose=False)
    analyzer.dropna(include_vars=["bool_obj_col"])

    emitter = analyzer.datahandler().train_test_emitter(
        y_var="target", X_vars=["bool_obj_col"]
    )
    X_train, _, X_test, _ = emitter.emit_train_test_Xy(verbose=False)

    # one-hot columns should use lowercase "true"/"false", not mixed casing
    onehot_cols = X_train.columns.tolist()
    for col in onehot_cols:
        assert (
            "True" not in col and "False" not in col
        ), f"One-hot column '{col}' has capitalized bool — mixed casing bug"

    # with dropfirst=True and 2 categories, expect exactly 1 column
    assert len(onehot_cols) == 1, (
        f"Expected 1 one-hot column (2 categories, drop first), "
        f"got {len(onehot_cols)}: {onehot_cols}"
    )


def test_csv_round_trip_true_false_handling():
    """Tests that data round-tripped through CSV (which converts TRUE/FALSE
    to Python bools) is properly normalized by the Analyzer."""
    import tempfile
    import os

    # create a CSV with TRUE/FALSE (which pd.read_csv will parse as bools)
    csv_content = "category,value\nTRUE,1\nFALSE,2\nTRUE,3\nFALSE,4\nTRUE,5\nFALSE,6\nTRUE,7\nFALSE,8\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        df = pd.read_csv(tmp_path)
        # pd.read_csv converts TRUE/FALSE to Python bool
        assert df["category"].dtype == bool

        analyzer = tm.Analyzer(df, test_size=0.3, verbose=False)
        all_df = analyzer.datahandler().df_all()

        # bool columns should be converted to int, not have mixed True/true
        assert all_df["category"].dtype in (np.int64, np.int32, int)
        assert set(all_df["category"].unique()).issubset({0, 1})
    finally:
        os.unlink(tmp_path)
