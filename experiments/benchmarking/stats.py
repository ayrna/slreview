import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def create_tukey_homogeneous_groups_table(
    endog, groups, alpha=0.05, return_tukey=False
):
    df = pd.DataFrame({"endog": endog, "groups": groups})
    means = df.groupby("groups")["endog"].mean().sort_values()

    # Tukey HSD test
    tukey = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=alpha)
    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:], columns=tukey._results_table.data[0]
    )

    # Unique sorted groups and index mapping
    unique_groups = means.index.tolist()
    idx_map = {g: i for i, g in enumerate(unique_groups)}
    n = len(unique_groups)
    adj = np.eye(n, dtype=bool)

    # Populate adjacency and p-value dict
    pval_dict = {}
    for _, row in tukey_df.iterrows():
        g1, g2 = row["group1"], row["group2"]
        i, j = idx_map[g1], idx_map[g2]
        pval = row["p-adj"]
        if not row["reject"]:
            adj[i, j] = adj[j, i] = True
        pval_dict[frozenset((g1, g2))] = pval

    # Find homogeneous subsets
    def find_subsets(adj, labels):
        subsets = []
        for size in range(n, 0, -1):
            for comb in combinations(range(n), size):
                if all(adj[i, j] for i, j in combinations(comb, 2)):
                    subset = set(labels[i] for i in comb)
                    if not any(subset <= s for s in subsets):
                        subsets.append(subset)
        return subsets

    subsets = find_subsets(adj, unique_groups)

    # Get min p-value for each subset
    subset_results = []
    for subset in subsets:
        pairs = combinations(sorted(subset), 2)
        min_p = min(
            (pval_dict.get(frozenset(pair), 1.0) for pair in pairs), default=1.0
        )
        subset_results.append((sorted(subset), min_p))

    # Display subsets
    # for i, (subset, min_p) in enumerate(subset_results, 1):
    #     print(f"Group {i}: {', '.join(subset)} (min p-value: {min_p:.4f})")

    # Build table
    columns = ["Group"] + [f"S{i+1}" for i in range(len(subsets))]
    data = {col: [] for col in columns}
    for g in unique_groups:
        data["Group"].append(g)
        for i, subset in enumerate(subsets):
            data[f"S{i+1}"].append(means[g] if g in subset else np.nan)
    # Add min p-value row
    data["Group"].append("min p-value")
    for i, (_, min_p) in enumerate(subset_results):
        data[f"S{i+1}"].append(min_p)

    result_df = pd.DataFrame(data)

    # Reorder subset columns so that groups are consecutively grouped
    subset_columns = []
    for _, row in result_df.iterrows():
        for col in result_df.columns[1:]:
            if pd.notna(row[col]) and col not in subset_columns:
                subset_columns.append(col)
    result_df = result_df[["Group"] + subset_columns]

    # Ensure column names match
    result_df.columns = ["Group"] + [f"S{i+1}" for i in range(len(subset_columns))]

    # Set index to Group
    result_df.set_index("Group", inplace=True)

    if return_tukey:
        return result_df, tukey
    return result_df
