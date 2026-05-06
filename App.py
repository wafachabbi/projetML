import joblib

model      = joblib.load("models/dso1_classifier.pkl")
feat_cols  = joblib.load("models/dso1_feature_cols.pkl")
threshold  = joblib.load("models/dso1_threshold.pkl")

kmeans     = joblib.load("models/dso2_kmeans.pkl")
scaler     = joblib.load("models/dso2_scaler.pkl")

tfidf      = joblib.load("models/dso3_tfidf.pkl")
knn        = joblib.load("models/dso3_knn.pkl")
matrix     = joblib.load("models/dso3_tfidf_matrix.pkl")