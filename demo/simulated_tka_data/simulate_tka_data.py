"""Simulates patient-level data from summary statistics in supp_table_4.md.

Generates a DataFrame with 1419 patients across three groups:
TKA (n=471), UKA (n=324), and Other (n=624).
Continuous variables are drawn from normal distributions parameterized
by group-specific means and SDs. Categorical variables are sampled
according to group-specific proportions. Missing values are introduced
at the rates specified in the table.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_TOTAL = 1419
N_TKA = 471
N_UKA = 324
N_OTHER = N_TOTAL - N_TKA - N_UKA


def _back_calculate(overall, tka, uka):
    """Back-calculate the Other group mean from overall and subgroup means."""
    return (N_TOTAL * overall - N_TKA * tka - N_UKA * uka) / N_OTHER


CONTINUOUS_VARS = [
    ("Age", 7, (67.3, 8.6), (67.5, 8.8), (66.8, 8.3)),
    ("Height", 1163, (1.7, 0.1), (1.7, 0.1), (1.7, 0.1)),
    ("Weight", 107, (85.7, 17.3), (84.9, 17.9), (86.7, 17.1)),
    ("Sulcus Angle", 84, (135.6, 8.7), (135.7, 8.5), (135.9, 8.3)),
    ("Trochlear Groove Width", 84, (36.7, 4.5), (36.3, 4.5), (36.6, 4.2)),
    ("Posterior Condylar Angle", 84, (7.4, 5.7), (7.5, 6.0), (7.2, 5.7)),
    ("TEA Horizontal Angle", 84, (7.4, 5.8), (7.7, 6.0), (7.7, 5.9)),
    ("TEA-PCA Angle", 84, (5.0, 2.3), (5.1, 2.3), (5.0, 2.3)),
    ("TT-TG Distance", 87, (12.0, 4.9), (12.6, 5.2), (11.6, 4.6)),
    ("Medial Posterior Tibial Slope", 84, (4.9, 4.3), (5.2, 4.6), (4.8, 4.0)),
    ("Lateral Posterior Tibial Slope", 84, (4.4, 4.3), (4.0, 3.9), (3.6, 3.7)),
    ("AP Femur Distance", 87, (60.1, 5.7), (60.8, 5.6), (60.5, 4.8)),
    ("AP Tibial Distance", 85, (50.0, 5.0), (50.5, 5.0), (50.0, 4.8)),
    ("Knee Flexion Angle", 107, (7.8, 7.5), (7.8, 6.3), (7.4, 5.6)),
    ("Femoral Neck Angle", 99, (9.5, 9.6), (9.6, 9.2), (10.8, 9.1)),
    ("Neck-Shaft Angle", 80, (128.3, 4.5), (128.5, 4.3), (128.7, 4.7)),
    ("Femoral Anteversion", 106, (11.5, 9.4), (11.4, 9.4), (11.8, 9.7)),
    ("Femoral Torsion", 112, (12.5, 7.6), (12.6, 7.5), (12.8, 7.8)),
    ("Acetabular Angle", 92, (8.4, 7.5), (7.0, 7.4), (7.7, 7.1)),
    ("Femoral Head Width", 133, (44.2, 4.2), (44.1, 4.1), (44.5, 4.0)),
    ("Joint Line Convergence Angle", 104, (2.7, 1.8), (2.5, 1.8), (3.0, 1.7)),
    ("Hip-Knee-Ankle Angle", 85, (173.5, 4.2), (173.7, 4.2), (173.0, 4.3)),
    ("Lateral Distal Femoral Angle", 81, (87.9, 3.5), (87.6, 3.7), (88.4, 3.3)),
    ("Medial Proximal Tibial Angle", 84, (85.9, 3.8), (85.9, 4.0), (85.3, 3.5)),
    ("Tibiofemoral Angle", 79, (4.5, 4.0), (5.3, 4.5), (3.8, 3.6)),
    ("Patella Width", 89, (44.0, 4.2), (44.0, 4.1), (43.9, 4.1)),
    ("Medial Femoral Condyle Width", 88, (28.1, 3.0), (28.2, 3.1), (28.3, 2.9)),
    ("Lateral Femoral Condyle Width", 91, (29.0, 3.1), (28.9, 2.8), (28.8, 3.0)),
    ("Tibial Width", 93, (76.6, 6.5), (76.9, 6.6), (76.6, 6.1)),
    ("Medial Tibial Width", 92, (29.2, 3.4), (29.0, 3.3), (29.1, 3.2)),
    ("Lateral Tibial Width", 92, (29.9, 3.5), (30.0, 3.6), (29.5, 3.1)),
]

CATEGORICAL_VARS = [
    (
        "Laterality",
        0,
        {
            "Left": (50.0, 50.7, 46.0),
            "Right": (49.8, 49.3, 54.0),
        },
    ),
    (
        "External Rotation",
        0,
        {
            "False": (56.2, 58.2, 56.2),
            "None": (7.7, 1.5, 2.8),
            "True": (36.1, 40.3, 41.0),
        },
    ),
    (
        "Varus Alignment",
        0,
        {
            "FALSE": (25.3, 33.3, 17.3),
            "None": (5.5, 1.1, 0.9),
            "TRUE": (69.2, 65.6, 81.8),
        },
    ),
    (
        "Sex",
        0,
        {
            "F": (56.7, 59.9, 53.7),
            "M": (43.3, 40.1, 46.3),
        },
    ),
]


def _simulate_continuous(rng, n, mean, sd):
    return rng.normal(loc=mean, scale=sd, size=n)


def _simulate_categorical(rng, levels_probs, n):
    levels = list(levels_probs.keys())
    probs = np.array([levels_probs[l] for l in levels])
    probs = probs / probs.sum()  # normalize
    return rng.choice(levels, size=n, p=probs)


def _introduce_missing(rng, series, n_missing):
    if n_missing <= 0 or n_missing >= len(series):
        return series
    idx = rng.choice(series.index, size=n_missing, replace=False)
    series.loc[idx] = np.nan
    return series


def simulate(seed=SEED):
    rng = np.random.default_rng(seed)

    groups = ["TKA"] * N_TKA + ["UKA"] * N_UKA + ["Other"] * N_OTHER
    df = pd.DataFrame({"Group": groups})

    # continuous variables
    for name, n_missing, overall_stats, tka_stats, uka_stats in CONTINUOUS_VARS:
        other_mean = _back_calculate(overall_stats[0], tka_stats[0], uka_stats[0])
        other_sd = overall_stats[1]

        tka_vals = _simulate_continuous(rng, N_TKA, *tka_stats)
        uka_vals = _simulate_continuous(rng, N_UKA, *uka_stats)
        other_vals = _simulate_continuous(rng, N_OTHER, other_mean, other_sd)

        col = pd.Series(
            np.concatenate([tka_vals, uka_vals, other_vals]),
            index=df.index,
            name=name,
        )
        df[name] = _introduce_missing(rng, col, n_missing)

    # categorical variables
    for name, n_missing, level_probs in CATEGORICAL_VARS:
        tka_probs = {l: v[1] for l, v in level_probs.items()}
        uka_probs = {l: v[2] for l, v in level_probs.items()}

        other_probs = {}
        for l, v in level_probs.items():
            other_pct = (N_TOTAL * v[0] - N_TKA * v[1] - N_UKA * v[2]) / N_OTHER
            other_probs[l] = max(other_pct, 0.1)  # floor to avoid negatives

        tka_vals = _simulate_categorical(rng, tka_probs, N_TKA)
        uka_vals = _simulate_categorical(rng, uka_probs, N_UKA)
        other_vals = _simulate_categorical(rng, other_probs, N_OTHER)

        col = pd.Series(
            np.concatenate([tka_vals, uka_vals, other_vals]),
            index=df.index,
            name=name,
        )
        if n_missing > 0:
            col = _introduce_missing(rng, col, n_missing)
        df[name] = col
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = simulate()
    out_path = Path(__file__).parent / "simulated_tka_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"\nGroup counts:\n{df['Group'].value_counts().to_string()}")
    print(f"\nFirst few rows:\n{df.head()}")
