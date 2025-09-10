# src/day1_netflix_clean.py
from pathlib import Path
import sys
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CSV_PATH = DATA / "netflix_titles.csv"

OUT.mkdir(exist_ok=True, parents=True)

# ---------- Load ----------
if not CSV_PATH.exists():
    sys.exit(f"[!] Missing dataset: {CSV_PATH}\nPut your CSV in the data/ folder and re-run.")

df = pd.read_csv(CSV_PATH)

print("\n=== HEAD ===")
print(df.head(3))
print("\n=== SHAPE ===", df.shape)

# ---------- Inspect & Clean ----------
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace(r"[^a-z0-9_]", "", regex=True)
)

# Basic NA summary
na_summary = df.isna().sum().sort_values(ascending=False)
print("\n=== MISSING VALUES (top 10) ===")
print(na_summary.head(10))

# Drop exact duplicate rows (if any)
df = df.drop_duplicates()

# Safe type conversions
if "date_added" in df.columns:
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

# ---------- Filter: Movies since 2015 ----------
filtered = df.copy()
if "type" in filtered.columns:
    filtered = filtered[filtered["type"].str.lower() == "movie"]

if "release_year" in filtered.columns:
    filtered = filtered[pd.to_numeric(filtered["release_year"], errors="coerce") >= 2015]

filtered = filtered.dropna(subset=["title"])
print("\n=== FILTERED SHAPE (Movies >= 2015) ===", filtered.shape)

# ---------- Genre analysis ----------
# Netflix genres are in 'listed_in' comma-separated
top_genres = None
if "listed_in" in filtered.columns:
    # one-hot encode genres and sum
    genre_dummies = filtered["listed_in"].str.get_dummies(sep=", ")
    genre_counts = genre_dummies.sum().sort_values(ascending=False)
    top_genres = genre_counts.head(10)
    print("\n=== TOP 10 GENRES (Movies >= 2015) ===")
    print(top_genres)

    # Plot & save
    plt.figure()
    top_genres.sort_values().plot(kind="barh")
    plt.title("Top Genres (Movies ≥ 2015)")
    plt.xlabel("Count")
    plt.tight_layout()
    plot_path = OUT / "top_genres_movies_2015plus.png"
    plt.savefig(plot_path)
    print(f"\n[+] Saved plot → {plot_path}")

# ---------- Basic KPIs ----------
kpis = {
    "total_rows_original": int(df.shape[0]),
    "total_rows_filtered": int(filtered.shape[0]),
    "unique_countries": int(filtered["country"].nunique()) if "country" in filtered.columns else None,
    "year_min": int(filtered["release_year"].min()) if "release_year" in filtered.columns and not filtered["release_year"].isna().all() else None,
    "year_max": int(filtered["release_year"].max()) if "release_year" in filtered.columns and not filtered["release_year"].isna().all() else None,
}
print("\n=== KPIs ===")
for k, v in kpis.items():
    print(f"{k}: {v}")

# ---------- Save cleaned subset ----------
clean_path = OUT / "netflix_movies_2015plus_clean.csv"
filtered.to_csv(clean_path, index=False)
print(f"\n[+] Saved cleaned data → {clean_path}")

# ---------- Optional: export quick summary ----------
summary_lines = [
    "# Netflix Movies (≥2015) — Day 1 Summary",
    f"- Original rows: {kpis['total_rows_original']}",
    f"- Filtered rows: {kpis['total_rows_filtered']}",
    f"- Year range: {kpis['year_min']}–{kpis['year_max']}",
    f"- Unique countries: {kpis['unique_countries']}",
]
if top_genres is not None:
    summary_lines.append("\nTop 10 Genres:\n" + "\n".join([f"- {g}: {int(c)}" for g, c in top_genres.items()]))

(OUT / "day1_summary.txt").write_text(
    "\n".join([s for s in summary_lines if s]),
    encoding="utf-8"
)
print(f"[+] Wrote summary → {OUT / 'day1_summary.txt'}")
