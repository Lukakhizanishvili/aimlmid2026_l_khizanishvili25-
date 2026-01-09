"""
AI and ML for Cybersecurity — Midterm Exam (Jan 9, 2026)
Luka Khizanishvili

This single-file script covers BOTH tasks:
1) Pearson correlation from manually captured graph data points
2) Spam email detection using Logistic Regression (70/30 split), evaluation, coefficients, and feature extraction for new email text.

How to run examples:
- Correlation:
  python midterm.py correlation --points correlation/data_points.csv --out correlation/scatter.png

- Train + evaluate spam model + visualizations:
  python midterm.py spam --data spam_classifier/email_features.csv --outdir spam_classifier

- Classify a custom email text using saved model:
  python midterm.py classify --model spam_classifier/model.joblib --columns spam_classifier/columns.json --email "FREE prize!!! click now"
"""


import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from joblib import dump, load


# -----------------------------
# Task 1: Correlation
# -----------------------------
def run_correlation(points_csv: str, out_path: str | None) -> float:
    df = pd.read_csv(points_csv)

    # Accept either x/y columns or first two columns
    if {"x", "y"}.issubset(df.columns):
        x = df["x"].astype(float)
        y = df["y"].astype(float)
    else:
        if df.shape[1] < 2:
            raise ValueError("Correlation CSV must have at least two columns (x,y).")
        x = df.iloc[:, 0].astype(float)
        y = df.iloc[:, 1].astype(float)

    r = float(pd.Series(x).corr(pd.Series(y), method="pearson"))

    # Plot scatter + simple trend line
    plt.figure()
    plt.scatter(x, y, label="Data points")
    # Trend line
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(float(x.min()), float(x.max()), 100)
    yy = m * xx + b
    plt.plot(xx, yy, label="Trend line")
    plt.title(f"Scatter Plot (Pearson r = {r:.4f})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[OK] Saved scatter plot to: {out_path}")
    else:
        plt.show()

    print(f"Pearson Correlation Coefficient (r): {r:.6f}")
    return r


# -----------------------------
# Task 2: Spam Detection
# -----------------------------

@dataclass
class SpambaseLikeExtractor:
    """
    Best-effort extractor for classic spam datasets similar to UCI Spambase:
    - word_freq_xxx  => percentage of word occurrences
    - char_freq_$    => percentage of character occurrences
    - capital_run_length_average/longest/total
    If your dataset columns differ, the script will still run; unknown features become 0.
    """

    feature_names: List[str]

    _word_freq_re = re.compile(r"^word_freq_(.+)$")
    _char_freq_re = re.compile(r"^char_freq_(.+)$")

    def extract(self, text: str) -> np.ndarray:
        text_l = text.lower()
        tokens = re.findall(r"[a-z0-9']+", text_l)
        total_words = max(len(tokens), 1)

        # For char freq, use raw lower text
        total_chars = max(len(text_l), 1)

        # Capital runs: consecutive uppercase sequences in original text
        caps_runs = [len(m.group(0)) for m in re.finditer(r"[A-Z]{2,}", text)]
        cap_total = sum(caps_runs)
        cap_longest = max(caps_runs) if caps_runs else 0
        cap_avg = (cap_total / len(caps_runs)) if caps_runs else 0.0

        # Precompute counts
        word_counts: Dict[str, int] = {}
        for t in tokens:
            word_counts[t] = word_counts.get(t, 0) + 1

        # char counts
        # For char_freq_ columns, the suffix might be '$', '!', '(', etc.
        def char_count(c: str) -> int:
            if len(c) == 1:
                return text_l.count(c)
            # Some datasets encode special chars like "semicolon" but we can't guess.
            return 0

        feats = []
        for name in self.feature_names:
            m1 = self._word_freq_re.match(name)
            if m1:
                w = m1.group(1)
                # Many datasets use words like "make", "address", "free", etc.
                cnt = word_counts.get(w.lower(), 0)
                # Spambase uses percentage of words in the email
                feats.append((cnt / total_words) * 100.0)
                continue

            m2 = self._char_freq_re.match(name)
            if m2:
                c = m2.group(1)
                # handle common escaping like "char_freq_$"
                # if suffix is something like "3", it's ambiguous => 0
                cnt = char_count(c)
                feats.append((cnt / total_chars) * 100.0)
                continue

            # Capital-run features (classic names)
            if name == "capital_run_length_average":
                feats.append(float(cap_avg))
                continue
            if name == "capital_run_length_longest":
                feats.append(float(cap_longest))
                continue
            if name == "capital_run_length_total":
                feats.append(float(cap_total))
                continue

            # Unknown feature — set 0 (but keep consistent dimension)
            feats.append(0.0)

        return np.array(feats, dtype=float)


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    # The exam says classes are spam or legitimate (sometimes encoded 1/0).
    # We’ll expect a column named 'class' (as stated in exam),
    # but also support common alternatives.
    class_col_candidates = ["class", "label", "target", "spam"]
    class_col = None
    for c in class_col_candidates:
        if c in df.columns:
            class_col = c
            break
    if class_col is None:
        raise ValueError(
            f"Could not find class column. Tried: {class_col_candidates}. "
            f"Found columns: {list(df.columns)}"
        )

    y = df[class_col]
    X = df.drop(columns=[class_col])

    # Convert y to numeric 0/1 if needed
    # Accept: 'spam'/'legitimate', 'spam'/'ham', True/False, 1/0, etc.
    if y.dtype == object:
        y_norm = y.astype(str).str.lower().str.strip()
        y = y_norm.map(
            {
                "spam": 1,
                "legitimate": 0,
                "ham": 0,
                "not spam": 0,
                "0": 0,
                "1": 1,
            }
        )
        if y.isna().any():
            # fallback: try to coerce numeric
            y = pd.to_numeric(y_norm, errors="coerce")
    y = y.astype(int)

    # Make X numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X, y


def train_and_evaluate_spam(csv_path: str, outdir: str) -> Dict[str, object]:
    os.makedirs(outdir, exist_ok=True)

    X, y = load_dataset(csv_path)

    # 70% training, 30% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y if y.nunique() == 2 else None
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Coefficients
    coefs = pd.DataFrame(
        {"feature": X.columns, "coefficient": model.coef_[0]}
    ).sort_values("coefficient", ascending=False)

    # Save artifacts for later classification
    model_path = os.path.join(outdir, "model.joblib")
    dump(model, model_path)

    columns_path = os.path.join(outdir, "columns.json")
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, ensure_ascii=False, indent=2)

    # --- Visualization 1: Class distribution ---
    plt.figure()
    counts = pd.Series(y).value_counts().sort_index()
    # Ensure labels: 0=Legitimate, 1=Spam
    x_labels = ["Legitimate(0)", "Spam(1)"] if set(counts.index) <= {0, 1} else [str(i) for i in counts.index]
    plt.bar(range(len(counts)), counts.values, tick_label=x_labels)
    plt.title("Class Distribution (Spam vs Legitimate)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    class_dist_path = os.path.join(outdir, "class_distribution.png")
    plt.grid(True, axis="y")
    plt.savefig(class_dist_path, dpi=200, bbox_inches="tight")

    # --- Visualization 2: Confusion Matrix heatmap (matplotlib only, no seaborn requirement) ---
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Legitimate(0)", "Spam(1)"])
    plt.yticks([0, 1], ["Legitimate(0)", "Spam(1)"])

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    cm_path = os.path.join(outdir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")

    # --- Optional Visualization 3: Top coefficients bar chart (useful for feature importance) ---
    topn = 15 if len(coefs) >= 15 else len(coefs)
    top_pos = coefs.head(topn).iloc[::-1]  # reverse for nice plotting

    plt.figure()
    plt.barh(top_pos["feature"], top_pos["coefficient"])
    plt.title(f"Top {topn} Positive Logistic Regression Coefficients (Spam-leaning)")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    coef_path = os.path.join(outdir, "top_coefficients.png")
    plt.grid(True, axis="x")
    plt.savefig(coef_path, dpi=200, bbox_inches="tight")

    # Print key outputs
    print("\n[RESULT] Confusion Matrix:\n", cm)
    print("\n[RESULT] Accuracy:", acc)
    print("\n[RESULT] Top coefficients (spam-leaning):")
    print(coefs.head(15).to_string(index=False))
    print("\n[OK] Saved model to:", model_path)
    print("[OK] Saved columns to:", columns_path)
    print("[OK] Saved plots:")
    print("  -", class_dist_path)
    print("  -", cm_path)
    print("  -", coef_path)

    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": float(acc),
        "model_path": model_path,
        "columns_path": columns_path,
        "plots": [class_dist_path, cm_path, coef_path],
        "top_coefficients": coefs.head(15).to_dict(orient="records"),
    }


def classify_email_text(model_path: str, columns_path: str, email_text: str) -> Tuple[int, float]:
    model = load(model_path)
    with open(columns_path, "r", encoding="utf-8") as f:
        cols = json.load(f)

    extractor = SpambaseLikeExtractor(feature_names=list(cols))
    feats = extractor.extract(email_text)

    # Shape to (1, n_features)
    X_new = feats.reshape(1, -1)
    pred = int(model.predict(X_new)[0])

    # probability if available
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_new)[0][1])  # prob of spam class 1
    else:
        proba = float("nan")

    label = "Spam" if pred == 1 else "Legitimate"
    print(f"[PREDICTION] {label} (class={pred}) | spam_probability={proba:.4f}")
    return pred, proba


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Midterm Exam single-file solution.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_corr = sub.add_parser("correlation", help="Compute Pearson correlation and plot scatter.")
    p_corr.add_argument("--points", required=True, help="CSV with x,y points (manually collected).")
    p_corr.add_argument("--out", default=None, help="Path to save plot (e.g., correlation/scatter.png).")

    p_spam = sub.add_parser("spam", help="Train + evaluate logistic regression spam classifier + plots.")
    p_spam.add_argument("--data", required=True, help="CSV dataset with features + class label.")
    p_spam.add_argument("--outdir", required=True, help="Output directory to save model and plots.")

    p_cls = sub.add_parser("classify", help="Classify a new email text using a saved model.")
    p_cls.add_argument("--model", required=True, help="Path to saved model.joblib")
    p_cls.add_argument("--columns", required=True, help="Path to saved columns.json")
    p_cls.add_argument("--email", required=True, help="Email text to classify")

    args = parser.parse_args()

    if args.cmd == "correlation":
        run_correlation(args.points, args.out)

    elif args.cmd == "spam":
        train_and_evaluate_spam(args.data, args.outdir)

    elif args.cmd == "classify":
        classify_email_text(args.model, args.columns, args.email)


if __name__ == "__main__":
    main()
