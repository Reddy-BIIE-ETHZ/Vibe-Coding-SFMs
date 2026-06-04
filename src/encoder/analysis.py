"""Script to collect and summarize cross-validation results from fold subdirectories."""

import argparse
import os

import pandas as pd


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder summary metric columns for consistent output.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing fold/phase metadata and evaluation metrics.

    Returns
    -------
    pandas.DataFrame
        DataFrame with preferred metric columns moved to the front while
        preserving any remaining columns.
    """
    k_values = [1, 5, 10, 30, 50]
    col_list = df.columns.tolist()

    col_order = []

    def append_if_exists(columns: list[str]) -> None:
        """Append column names to col_order if they exist in col_list."""
        for col in columns:
            if col in col_list:
                col_order.append(col)

    append_if_exists(["fold", "phase", "epoch", "samples", "batch_count"])

    for k in k_values:
        recall_col_ag = f"recall_at_{k}_ag"
        recall_col_ab = f"recall_at_{k}_ab"
        if recall_col_ag in col_list:
            col_order.append(recall_col_ag)
        if recall_col_ab in col_list:
            col_order.append(recall_col_ab)
    for k in k_values:
        precision_col_ag = f"precision_at_{k}_ag"
        precision_col_ab = f"precision_at_{k}_ab"
        if precision_col_ag in col_list:
            col_order.append(precision_col_ag)
        if precision_col_ab in col_list:
            col_order.append(precision_col_ab)
    for k in k_values:
        recall_col = f"recall_at_{k}"
        if recall_col in col_list:
            col_order.append(recall_col)
    for k in k_values:
        precision_col = f"precision_at_{k}"
        if precision_col in col_list:
            col_order.append(precision_col)

    append_if_exists(
        [
            "ef1_ag2ab",
            "ef1_ab2ag",
            "ef1",
            "avg_cosine_sim",
            "avg_margin_max_neg_ag",
            "avg_margin_max_neg_ab",
            "mrr_ag2ab",
            "mrr_ab2ag",
            "mrr_avg_rank_ag2ab",
            "mrr_avg_rank_ab2ag",
            "mrr_max_rank_ag2ab",
            "mrr_max_rank_ab2ag",
            "mrr",
        ]
    )

    col_remaining = [col for col in col_list if col not in col_order]
    col_order.extend(col_remaining)
    return df[col_order]


def collect_results(dir_in: str, mode: str = "test", save_results: bool = True) -> None:
    """Collect best-epoch cross-validation results and summarize by fold.

    Parameters
    ----------
    dir_in : str
        Input directory containing fold subdirectories, or a parent directory
        that contains a ``train`` folder.
    mode : {"val", "test"}, default="test"
        Evaluation split to summarize.
    save_results : bool, default=True
        If True, write the summary CSV to disk.

    Returns
    -------
    None
        Prints summary statistics and optionally saves a CSV file.
    """
    results = []
    fold_list = []

    if os.path.basename(dir_in) != "train":
        child_folder = os.listdir(dir_in)
        if "train" in child_folder:
            dir_in = os.path.join(dir_in, "train")
        else:
            raise FileNotFoundError(f"'train' folder not found in {dir_in}")

    folder_fold_list = [
        f for f in os.listdir(dir_in) if os.path.isdir(os.path.join(dir_in, f))
    ]
    for folder_fold in folder_fold_list:
        dir_fold = os.path.join(dir_in, folder_fold)
        if mode == "val":
            file_results = os.path.join(dir_fold, "results_train_val_test.csv")
        elif mode == "test":
            file_results = os.path.join(dir_fold, "results_train_test.csv")

        if os.path.exists(file_results):
            df = pd.read_csv(file_results)

            if mode == "val":
                best_epoch = int(
                    df[df["phase"] == mode]
                    .sort_values(by=["pred_acc", "epoch"], ascending=[False, True])
                    .iloc[0]["epoch"]
                )

            elif mode == "test":
                file_final_model = [  # noqa: RUF015
                    f for f in os.listdir(dir_fold) if "final_model" in f
                ][0]
                best_epoch = int(file_final_model.split("_")[-1].split(".")[0])

            df_best_row = df[(df["phase"] == mode) & (df["epoch"] == best_epoch)]

            results.append(df_best_row)
            fold_list.append(folder_fold)

    df_summary = pd.concat(results, axis=0)
    df_summary["fold"] = fold_list

    # Add mean and std rows for all columns except 'fold'
    numeric_cols = df_summary.select_dtypes(include=["number"]).columns
    mean_row = df_summary[numeric_cols].mean()
    std_row = df_summary[numeric_cols].std()

    # Create new rows with 'fold' set to 'fold_avg' and 'fold_std'
    mean_row_full = {col: mean_row.get(col, None) for col in df_summary.columns}
    mean_row_full["fold"] = "Avg"
    std_row_full = {col: std_row.get(col, None) for col in df_summary.columns}
    std_row_full["fold"] = "Std"

    # Append to df_summary
    df_summary = pd.concat(
        [
            df_summary,
            pd.DataFrame([mean_row_full, std_row_full], columns=df_summary.columns),
        ],
        ignore_index=True,
    )

    # column reordering
    df_summary = reorder_columns(df_summary)

    if save_results:
        file_out = os.path.join(dir_in, f"summary_results_{mode}.csv")
        df_summary.to_csv(file_out, index=False)
        print(f"Saved summary to {file_out}")

    # Mean/median of CV
    average_cv_best_acc = round(
        df_summary[df_summary["fold"].str.contains("fold")]["pred_acc"].mean(), 3
    )
    sd_cv_best_acc = round(
        df_summary[df_summary["fold"].str.contains("fold")]["pred_acc"].std(), 3
    )
    average_cv_best_epoch = int(
        round(df_summary[df_summary["fold"].str.contains("fold")]["epoch"].mean())
    )
    median_cv_best_epoch = int(
        round(df_summary[df_summary["fold"].str.contains("fold")]["epoch"].median())
    )
    print(f"Results for mode: {mode}")
    print(f"Average CV Best Accuracy: {average_cv_best_acc}")
    print(f"SD CV Best Accuracy: {sd_cv_best_acc}")
    print(f"Average CV Best Epoch: {average_cv_best_epoch}")
    print(f"Median CV Best Epoch: {median_cv_best_epoch}")
    print("")


def main() -> None:
    """Parse CLI arguments and run result collection for val and test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_in", type=str, required=True)

    args = parser.parse_args()
    dir_in = args.dir_in

    eval_mode = ["val", "test"]
    for mode in eval_mode:
        collect_results(dir_in, mode=mode)


if __name__ == "__main__":
    main()
