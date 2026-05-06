"""
save_models.py
==============
Entraîne et sauvegarde tous les modèles du projet CRISP-DM Udemy en fichiers .pkl.

Modèles générés :
  - models/dso1_classifier.pkl       → XGBoost (ou RandomForest fallback)
  - models/dso1_feature_cols.pkl     → liste des colonnes features
  - models/dso1_threshold.pkl        → seuil de popularité (médiane)
  - models/dso2_kmeans.pkl           → KMeans (clustering optimal)
  - models/dso2_scaler.pkl           → StandardScaler (clustering)
  - models/dso2_pca.pkl              → PCA 2D (visualisation)
  - models/dso3_tfidf.pkl            → TfidfVectorizer (recommandation)
  - models/dso3_knn.pkl              → NearestNeighbors (recommandation)
  - models/dso3_tfidf_matrix.pkl     → matrice TF-IDF sparse
  - models/dso3_course_titles.pkl    → index des titres de cours

Usage :
    python save_models.py
    # ou en précisant le chemin du CSV :
    python save_models.py --data udemy_courses.csv
"""

import sys
import os
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Dossier de sortie ─────────────────────────────────────────────────────────
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def save(obj, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    joblib.dump(obj, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"   ✅  {filename:<40}  ({size_kb:.1f} KB)")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Capping IQR + features dérivées — identique à App.py."""
    df2 = df.copy()
    for col in ["price", "num_subscribers", "num_reviews", "num_lectures", "content_duration"]:
        Q1, Q3 = df2[col].quantile(0.25), df2[col].quantile(0.75)
        df2[col] = df2[col].clip(Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1))
    df2["reviews_per_sub"] = (
        df2["num_reviews"] / df2["num_subscribers"].replace(0, np.nan)
    ).fillna(0)
    df2["lectures_per_hour"] = (
        df2["num_lectures"] / df2["content_duration"].replace(0, np.nan)
    ).fillna(0)
    return df2


# =============================================================================
# DSO 1 — Classification (XGBoost, fallback RandomForest)
# =============================================================================

def train_dso1(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("🎯  DSO 1 — Classification (prédiction popularité)")
    print("=" * 65)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.impute import SimpleImputer

    df2 = prepare_features(df)
    threshold = float(df2["num_subscribers"].median())
    y = (df2["num_subscribers"] >= threshold).astype(int)

    drop_cols = ["course_id", "url", "published_timestamp", "course_title", "num_subscribers"]
    X = df2.drop(columns=[c for c in drop_cols if c in df2.columns])
    X = pd.get_dummies(X, columns=["subject", "level"], drop_first=True)
    X["is_paid"] = X["is_paid"].astype(int)

    # Imputation des NaN résiduels
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    Xtr, Xte, ytr, yte = train_test_split(X_imp, y, test_size=0.2, random_state=42)

    # ── Essai XGBoost ────────────────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0
        )
        print("   Algorithme : XGBClassifier")
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        print("   ⚠️  XGBoost non disponible → RandomForestClassifier utilisé")

    model.fit(Xtr, ytr)
    yp  = model.predict(Xte)
    ypr = model.predict_proba(Xte)[:, 1]

    print(f"   Accuracy  : {accuracy_score(yte, yp):.4f}")
    print(f"   F1-Score  : {f1_score(yte, yp):.4f}")
    print(f"   AUC-ROC   : {roc_auc_score(yte, ypr):.4f}")
    print(f"   Seuil pop : {threshold:,.0f} inscrits")
    print(f"\n   Sauvegarde ...")

    feat_cols = X_imp.columns.tolist()
    save(model,     "dso1_classifier.pkl")
    save(feat_cols, "dso1_feature_cols.pkl")
    save(threshold, "dso1_threshold.pkl")


# =============================================================================
# DSO 2 — Clustering (KMeans optimal)
# =============================================================================

def train_dso2(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("🔵  DSO 2 — Clustering (KMeans)")
    print("=" * 65)

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.impute import SimpleImputer

    df2 = prepare_features(df)
    feats = ["price", "num_reviews", "num_lectures", "content_duration",
             "reviews_per_sub", "lectures_per_hour"]
    feats = [f for f in feats if f in df2.columns]

    X = df2[feats].copy()
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)

    # Recherche du k optimal (silhouette)
    best_k, best_sil = 4, -1
    print("   Recherche k optimal ...")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        s  = silhouette_score(X_sc, km.fit_predict(X_sc))
        if s > best_sil:
            best_sil, best_k = s, k
        print(f"      k={k} | Silhouette={s:.4f}")

    print(f"\n   🏆 Meilleur k : {best_k}  (Silhouette={best_sil:.4f})")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_final.fit(X_sc)

    pca_2d = PCA(n_components=2, random_state=42)
    pca_2d.fit(X_sc)
    print(f"   PCA variance expliquée : {pca_2d.explained_variance_ratio_.sum():.1%}")

    print(f"\n   Sauvegarde ...")
    save(km_final, "dso2_kmeans.pkl")
    save(scaler,   "dso2_scaler.pkl")
    save(pca_2d,   "dso2_pca.pkl")
    save(feats,    "dso2_cluster_features.pkl")


# =============================================================================
# DSO 3 — Recommandation (TF-IDF + KNN cosine)
# =============================================================================

def train_dso3(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("📚  DSO 3 — Recommandation (TF-IDF + KNN cosine)")
    print("=" * 65)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    df2 = df.copy()
    df2["combined"] = (
        df2["course_title"].fillna("") + " " +
        df2["subject"].fillna("") + " " +
        df2["level"].fillna("")
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        lowercase=True
    )
    matrix = tfidf.fit_transform(df2["combined"])
    print(f"   Matrice TF-IDF : {matrix.shape[0]} cours × {matrix.shape[1]} features")

    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=11)
    knn.fit(matrix)
    print("   KNN entraîné avec succès")

    course_titles = df["course_title"].tolist()

    print(f"\n   Sauvegarde ...")
    save(tfidf,         "dso3_tfidf.pkl")
    save(knn,           "dso3_knn.pkl")
    save(matrix,        "dso3_tfidf_matrix.pkl")   # matrice sparse (~léger)
    save(course_titles, "dso3_course_titles.pkl")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entraîne et sauvegarde les modèles Udemy.")
    parser.add_argument("--data", default="udemy_courses.csv",
                        help="Chemin vers le fichier CSV (défaut: udemy_courses.csv)")
    args = parser.parse_args()

    print("=" * 65)
    print("🚀  save_models.py — Projet CRISP-DM Udemy")
    print("=" * 65)
    print(f"   Chargement : {args.data}")

    df = pd.read_csv(args.data)
    print(f"   Dataset     : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    train_dso1(df)
    train_dso2(df)
    train_dso3(df)

    print("\n" + "=" * 65)
    print(f"✅  Tous les modèles sont sauvegardés dans → ./{OUTPUT_DIR}/")
    print("=" * 65)
    files = sorted(os.listdir(OUTPUT_DIR))
    total_kb = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in files) / 1024
    for f in files:
        kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"   📦  {f:<42} {kb:7.1f} KB")
    print(f"\n   Total : {total_kb:.1f} KB")


if __name__ == "__main__":
    main()
