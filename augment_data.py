"""
augment_data.py  -  FuturePath case-study data generator
=========================================================

Builds the dataset for the "FuturePath" juvenile-rehabilitation audit from the
real ProPublica COMPAS file. It is fully DETERMINISTIC (single fixed seed), so
the notebook's results are reproducible: re-running this script always yields a
byte-identical `futurepath_augmented.csv`.

What it does
------------
1. Reads the real COMPAS rows (real age, priors_count, juvenile counts, label).
2. Applies the standard ProPublica data-quality filters.
3. Adds SYNTHETIC "social-circumstance" columns (ses_level, parental_stability,
   school_status, has_mentor, concrete_rehab_plan), correlated with the real
   signals so they are plausible rather than arbitrary.
4. Defines the true rehabilitation outcome as the inverse of re-offending.
5. Builds the deployed FuturePath "rehabilitation-success" decile (1-10) so that
   it leans on STRUCTURAL features (SES, parental stability, dropout) and IGNORES
   the changeable positives (mentor, concrete plan). That injected dependence is
   the injustice the notebook then audits.

NOTE (honesty): the social columns and the FuturePath score are synthetic. They
are NOT real measurements. This is a teaching case; the point is a reproducible
illustration of "punishing people for social circumstances", not an empirical
claim about any deployed system.
"""

import numpy as np
import pandas as pd

SEED = 42
SRC = "compas-scores-two-years_v1.csv"
OUT = "futurepath_augmented.csv"


def load_and_filter(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    # COMPAS ships two columns each named decile_score / priors_count; the second
    # of each pair is the duplicate. Keep deterministic first occurrence.
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()

    cols = [
        "age", "sex", "race", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "priors_count", "c_charge_degree",
        "score_text", "days_b_screening_arrest", "two_year_recid",
    ]
    df = raw[cols].copy()

    # Standard ProPublica data-quality filters.
    df = df[
        (df["days_b_screening_arrest"] >= -30)
        & (df["days_b_screening_arrest"] <= 30)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"].notna())
    ].reset_index(drop=True)
    return df


def add_social_columns(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    n = len(df)

    # ---- latent "structural disadvantage" -------------------------------
    # Built from REAL signals: more priors, more juvenile contacts and younger
    # age all push disadvantage up. (This mirrors how real tools entangle
    # disadvantage with "risk".) Standardise to mean 0, sd 1.
    juv = df["juv_fel_count"] + df["juv_misd_count"] + df["juv_other_count"]
    raw_disadv = (
        0.6 * _z(df["priors_count"])
        + 0.3 * _z(juv)
        - 0.4 * _z(df["age"])
    )
    disadv = _z(raw_disadv) + rng.normal(0, 0.4, n)   # add idiosyncratic noise

    # ---- ses_level (1 = lowest .. 5 = highest) --------------------------
    # Lower SES for higher disadvantage. Quintile split of (-disadv).
    ses = pd.qcut(-disadv, 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # ---- parental_stability (1 = stable home, 0 = unstable) -------------
    p_stable = _sigmoid(-0.9 * disadv + 0.3)
    parental_stability = (rng.rand(n) < p_stable).astype(int)

    # ---- school_status --------------------------------------------------
    p_dropout = _sigmoid(0.9 * disadv - 0.2)
    school_status = np.where(rng.rand(n) < p_dropout, "Dropout", "In school")

    df = df.assign(
        ses_level=ses.values,
        parental_stability=parental_stability,
        school_status=school_status,
        _disadv=disadv,
    )
    return df


def add_outcome_and_score(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    n = len(df)

    # True rehabilitation success = did NOT re-offend within two years (REAL label).
    df["rehab_success_true"] = (1 - df["two_year_recid"]).astype(int)

    # ---- changeable, RECENT positives -----------------------------------
    # These are facts a fair, current review SHOULD weigh: they are genuinely
    # INFORMATIVE (more common among youths who actually rehabilitate), yet the
    # deployed tool ignores them entirely. That gap is the "stale-data" harm.
    p_mentor = 0.15 + 0.35 * df["rehab_success_true"]
    p_plan = 0.15 + 0.35 * df["rehab_success_true"]
    df["has_mentor"] = (rng.rand(n) < p_mentor).astype(int)
    df["concrete_rehab_plan"] = (rng.rand(n) < p_plan).astype(int)

    # ---- Deployed FuturePath success score ------------------------------
    # Leans on STRUCTURAL features and only weakly on the true outcome; the
    # informative changeable positives (mentor, plan) are NOT used at all.
    # This is the injected injustice the audit detects.
    in_school = (df["school_status"] == "In school").astype(int)
    latent = (
        0.55 * _z(df["ses_level"])               # structural
        + 0.45 * (df["parental_stability"] - 0.5) * 2   # structural
        + 0.45 * (in_school - 0.5) * 2            # structural (changed for defendant X)
        + 0.50 * (df["rehab_success_true"] - 0.5) * 2   # weak signal of truth
        + rng.normal(0, 1.15, n)                 # noise
        # NOTE: has_mentor / concrete_rehab_plan intentionally absent.
    )
    df["success_decile"] = pd.qcut(latent, 10, labels=range(1, 11)).astype(int)
    df["success_text"] = pd.cut(
        df["success_decile"], bins=[0, 4, 7, 10],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    # Deployed decision rule: score <= 4 => "low rehabilitation potential" =>
    # diverted to a CLOSED institution. This is the harmful flag (analog of the
    # COMPAS high-risk flag).
    df["flag_low_potential"] = (df["success_decile"] <= 4).astype(int)
    return df.drop(columns=["_disadv"])


def _z(s):
    s = pd.Series(s, dtype="float64")
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    rng = np.random.RandomState(SEED)
    df = load_and_filter(SRC)
    df = add_social_columns(df, rng)
    df = add_outcome_and_score(df, rng)
    df.to_csv(OUT, index=False)

    # ---- quick reproducibility / sanity report --------------------------
    print(f"Rows written: {len(df):,}  ->  {OUT}")
    g = df.assign(
        ses_group=np.where(df.ses_level <= 2, "Low-SES",
                    np.where(df.ses_level >= 4, "High-SES", "Mid"))
    )
    g = g[g.ses_group != "Mid"]

    def _row(d):
        return pd.Series({
            "n": len(d),
            "true_rehab_rate": d.rehab_success_true.mean(),
            "flag_low_rate": d.flag_low_potential.mean(),
            # false "low potential": flagged low among those who DID rehabilitate
            "false_low_rate": d.loc[d.rehab_success_true == 1, "flag_low_potential"].mean(),
        })

    rep = g.groupby("ses_group")[g.columns].apply(_row).round(3)
    print(rep.to_string())


if __name__ == "__main__":
    main()
