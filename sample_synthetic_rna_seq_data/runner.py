"""This module is for executing all the different parts of the code in one place."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def runner():
    """All operations here."""

    # 1.Build data
    np.random.seed(42)

    n_genes = 100
    n_samples = 20

    # 1-1. Create expr matrix
    expr = np.random.poisson(lam=50, size=(n_genes, n_samples))

    # 1-2. DataFrame
    genes = [f"G{i}" for i in range(1, n_genes + 1)]
    samples = [f"S{i}" for i in range(1, n_samples + 1)]
    expr_df = pd.DataFrame(expr, index=genes, columns=samples)

    # 2. Create metadata
    conditions = ["A"] * 10 + ["B"] * 10
    meta_df = pd.DataFrame({"sample": samples, "condition": conditions})

    # 3. Connect expression to metadata
    # Transpose → نمونه‌ها بشن ردیف
    expr_t = expr_df.T
    expr_t["condition"] = meta_df["condition"].values

    # 4.Sample statistics analysis
    group_means = expr_t.groupby("condition").mean().T
    group_means["diff"] = group_means["B"] - group_means["A"]

    top_diff = group_means["diff"].abs().sort_values(ascending=False).head(10)

    # 5. Visualization
    genes_to_plot = list(top_diff.index)
    plot_df = expr_t[genes_to_plot + ["condition"]]

    melted = plot_df.melt(id_vars="condition", var_name="gene", value_name="expr")

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="gene", y="expr", hue="condition", data=melted)
    plt.title("Top Differential Genes")
    plt.show()

    # Scatter A vs B برای همه ژن‌ها
    plt.figure(figsize=(6, 6))
    plt.scatter(group_means["A"], group_means["B"], alpha=0.7)
    plt.xlabel("Mean Expr (Condition A)")
    plt.ylabel("Mean Expr (Condition B)")
    plt.title("Gene Expression Comparison A vs B")
    plt.plot([0, 100], [0, 100], "--", color="red")
    plt.show()
