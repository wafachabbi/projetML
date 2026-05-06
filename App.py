import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Smartek AI · Udemy Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#f9fafb; color:#111827; }
.stApp { background:#f9fafb; }
.block-container { padding:2rem 2.5rem; max-width:1200px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background:#1e1b4b;
    border-right:none;
    min-width:240px !important;
    max-width:240px !important;
}
[data-testid="stSidebar"] * { color:#c7d2fe !important; }
[data-testid="stSidebar"] hr { border-color:rgba(199,210,254,.1); }
[data-testid="collapsedControl"] { display:none !important; }
[data-testid="stSidebar"] .stRadio label {
    background:rgba(255,255,255,.05) !important;
    border:1px solid rgba(255,255,255,.08) !important;
    border-radius:8px !important;
    padding:.5rem .9rem !important;
    margin:.2rem 0 !important;
    color:#c7d2fe !important;
    font-size:.88rem !important;
    font-weight:500 !important;
    transition:all .15s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background:rgba(99,102,241,.2) !important;
    border-color:rgba(99,102,241,.4) !important;
    color:#fff !important;
}

/* Page header */
.page-header {
    margin-bottom:2rem;
    margin-top:1rem;
    padding:1.5rem 2rem;
    background:#fff;
    border-radius:14px;
    border:1px solid #e5e7eb;
    box-shadow:0 1px 4px rgba(0,0,0,.04);
}
.page-title {
    font-size:1.8rem; font-weight:800; color:#111827; line-height:1.2;
}
.page-sub { font-size:.9rem; color:#6b7280; margin-top:.4rem; }

/* KPI */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin:1.5rem 0; }
.kpi-card {
    background:#fff; border-radius:12px; padding:1.3rem 1.4rem;
    border:1px solid #e5e7eb;
    box-shadow:0 1px 4px rgba(0,0,0,.04);
    transition:box-shadow .2s;
}
.kpi-card:hover { box-shadow:0 4px 16px rgba(0,0,0,.08); }
.kpi-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:.8rem; }
.kpi-icon {
    width:38px; height:38px; border-radius:10px;
    display:flex; align-items:center; justify-content:center; font-size:1.1rem;
}
.kpi-badge {
    font-size:.68rem; font-weight:600; padding:.2rem .55rem;
    border-radius:99px; background:#f0fdf4; color:#16a34a;
}
.kpi-val { font-size:1.8rem; font-weight:800; color:#111827; line-height:1; }
.kpi-lbl { font-size:.72rem; color:#9ca3af; text-transform:uppercase; letter-spacing:1px; margin-top:.3rem; }

/* Section title */
.sec { font-size:.95rem; font-weight:700; color:#111827; margin:2rem 0 1rem; display:flex; align-items:center; gap:.6rem; }
.sec span { flex:1; height:1px; background:#e5e7eb; }

/* Cards */
.card {
    background:#fff; border-radius:12px; padding:1.4rem;
    border:1px solid #e5e7eb; box-shadow:0 1px 4px rgba(0,0,0,.04);
    margin-bottom:.8rem;
}

/* DSO cards */
.dso { background:#fff; border-radius:12px; padding:1.5rem; border:1px solid #e5e7eb; box-shadow:0 1px 4px rgba(0,0,0,.04); height:100%; transition:box-shadow .2s; }
.dso:hover { box-shadow:0 4px 16px rgba(0,0,0,.08); }
.dso-icon { font-size:1.8rem; margin-bottom:.7rem; }
.dso-title { font-size:.92rem; font-weight:700; color:#111827; margin-bottom:.4rem; }
.dso-desc { font-size:.82rem; color:#6b7280; line-height:1.65; }
.dso-chip {
    display:inline-block; margin-top:.8rem;
    background:#eef2ff; color:#4f46e5;
    font-size:.68rem; font-weight:600;
    padding:.2rem .65rem; border-radius:99px;
}

/* Prediction */
.res-yes { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:1.5rem; margin:.8rem 0; }
.res-no  { background:#fffbeb; border:1px solid #fde68a; border-radius:12px; padding:1.5rem; margin:.8rem 0; }
.res-title { font-size:1.2rem; font-weight:800; color:#111827; }
.res-score { font-size:2.2rem; font-weight:800; margin-top:.4rem; }

/* Rec */
.rec {
    background:#fff; border:1px solid #e5e7eb; border-radius:10px;
    padding:.9rem 1.1rem; margin:.4rem 0;
    display:flex; align-items:center; gap:.9rem;
    box-shadow:0 1px 3px rgba(0,0,0,.03);
    transition:border-color .15s;
}
.rec:hover { border-color:#a5b4fc; }
.rec-n { font-size:1.1rem; font-weight:800; color:#e0e7ff; min-width:2rem; }
.rec-t { font-size:.88rem; font-weight:600; color:#111827; }
.rec-m { font-size:.73rem; color:#9ca3af; margin-top:.15rem; }
.rec-p { font-size:.92rem; font-weight:700; color:#4f46e5; white-space:nowrap; }
.pb { background:#f3f4f6; border-radius:99px; height:3px; margin-top:.3rem; }
.pf { background:#4f46e5; height:3px; border-radius:99px; }

/* Button */
.stButton>button {
    background:#4f46e5 !important; color:#fff !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; font-size:.88rem !important;
    padding:.55rem 1.4rem !important;
    box-shadow:0 1px 4px rgba(79,70,229,.3) !important;
    transition:background .15s !important;
}
.stButton>button:hover { background:#4338ca !important; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

PAL = ['#4f46e5','#0891b2','#059669','#d97706','#dc2626','#7c3aed','#db2777']

@st.cache_data
def load_data():
    return pd.read_csv("udemy_courses.csv")

@st.cache_resource
def load_models():
    model     = joblib.load("models/dso1_classifier.pkl")
    feat_cols = joblib.load("models/dso1_feature_cols.pkl")
    threshold = joblib.load("models/dso1_threshold.pkl")
    kmeans    = joblib.load("models/dso2_kmeans.pkl")
    scaler    = joblib.load("models/dso2_scaler.pkl")
    tfidf     = joblib.load("models/dso3_tfidf.pkl")
    knn       = joblib.load("models/dso3_knn.pkl")
    matrix    = joblib.load("models/dso3_tfidf_matrix.pkl")
    return model, feat_cols, threshold, kmeans, scaler, tfidf, knn, matrix

@st.cache_data
def prepare_features(df):
    df2 = df.copy()
    for col in ['price','num_subscribers','num_reviews','num_lectures','content_duration']:
        Q1,Q3 = df2[col].quantile(.25), df2[col].quantile(.75)
        df2[col] = df2[col].clip(Q1-1.5*(Q3-Q1), Q3+1.5*(Q3-Q1))
    df2['reviews_per_sub']   = (df2['num_reviews']/df2['num_subscribers'].replace(0,np.nan)).fillna(0)
    df2['lectures_per_hour'] = (df2['num_lectures']/df2['content_duration'].replace(0,np.nan)).fillna(0)
    return df2

def fig(w=6, h=4):
    f, ax = plt.subplots(figsize=(w, h))
    f.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f9fafb')
    ax.tick_params(colors='#9ca3af', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#e5e7eb')
    ax.title.set_color('#111827')
    return f, ax

df = load_data()
model, feat_cols, threshold, kmeans, scaler, tfidf, knn_model, matrix = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.4rem 0 1rem;text-align:center;'>
        <div style='font-size:2rem;margin-bottom:.3rem;'>🧠</div>
        <div style='font-size:1.2rem;font-weight:800;color:#fff;'>Smartek AI</div>
        <div style='font-size:.6rem;color:rgba(199,210,254,.4);letter-spacing:2.5px;text-transform:uppercase;margin-top:.2rem;'>ML Dashboard</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "🏠  Vue Générale",
        "🎯  DSO 1 — Classification",
        "🔵  DSO 2 — Clustering",
        "📚  DSO 3 — Recommandation"
    ])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.73rem;color:rgba(199,210,254,.5);line-height:2.2;padding:.2rem .3rem;'>
        <div style='color:#818cf8;font-weight:700;margin-bottom:.2rem;'>Dataset</div>
        3 678 cours · 4 sujets<br><br>
        <div style='color:#818cf8;font-weight:700;margin-bottom:.2rem;'>Modèles</div>
        XGBoost · KMeans<br>TF-IDF + KNN<br><br>
        <div style='color:#818cf8;font-weight:700;margin-bottom:.2rem;'>Méthode</div>
        CRISP-DM
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — VUE GÉNÉRALE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Vue Générale":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🧠 Smartek AI Dashboard</div>
        <div class="page-sub">Analyse intelligente des cours Udemy · Classification · Clustering · Recommandation</div>
    </div>""", unsafe_allow_html=True)

    total=len(df); avg_sub=int(df['num_subscribers'].mean())
    free_pct=round((df['price']==0).mean()*100,1); avg_rev=int(df['num_reviews'].mean())

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-top"><div class="kpi-icon" style="background:#eef2ff;">🎓</div><div class="kpi-badge">Total</div></div>
            <div class="kpi-val">{total:,}</div><div class="kpi-lbl">Cours indexés</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top"><div class="kpi-icon" style="background:#ecfdf5;">👥</div><div class="kpi-badge">Moy.</div></div>
            <div class="kpi-val">{avg_sub:,}</div><div class="kpi-lbl">Inscrits moyens</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top"><div class="kpi-icon" style="background:#fefce8;">⭐</div><div class="kpi-badge">Moy.</div></div>
            <div class="kpi-val">{avg_rev:,}</div><div class="kpi-lbl">Avis moyens</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top"><div class="kpi-icon" style="background:#f0f9ff;">🆓</div><div class="kpi-badge">%</div></div>
            <div class="kpi-val">{free_pct}%</div><div class="kpi-lbl">Cours gratuits</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">Objectifs Data Science <span></span></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3, gap="medium")
    for col,icon,title,desc,chip in [
        (c1,"🎯","DSO 1 — Classification","Prédire si un cours sera populaire ou non à partir de ses caractéristiques.","XGBoost"),
        (c2,"🔵","DSO 2 — Clustering","Segmenter les cours en groupes homogènes via KMeans optimisé.","KMeans + Silhouette"),
        (c3,"📚","DSO 3 — Recommandation","Suggérer des formations similaires via TF-IDF + similarité cosinus.","TF-IDF + KNN"),
    ]:
        with col:
            st.markdown(f"""
            <div class="dso">
                <div class="dso-icon">{icon}</div>
                <div class="dso-title">{title}</div>
                <div class="dso-desc">{desc}</div>
                <div class="dso-chip">{chip}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">Aperçu du Dataset <span></span></div>', unsafe_allow_html=True)
    ca,cb,cc = st.columns(3, gap="medium")
    with ca:
        f,ax = fig(5,3.5)
        sc = df['subject'].value_counts()
        bars = ax.barh(sc.index, sc.values, color=PAL[:len(sc)], height=.5)
        ax.bar_label(bars, fontsize=8, padding=3, color='#6b7280')
        ax.set_title("Cours par Sujet", fontsize=10, fontweight='bold')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(f)
    with cb:
        f,ax = fig(5,3.5)
        lv = df['level'].value_counts()
        _,_,autotexts = ax.pie(lv.values, labels=lv.index, colors=PAL[:len(lv)],
            autopct='%1.0f%%', startangle=90,
            textprops={'fontsize':8,'color':'#374151'},
            wedgeprops={'edgecolor':'white','linewidth':2})
        for at in autotexts: at.set_fontweight('700')
        ax.set_title("Répartition par Niveau", fontsize=10, fontweight='bold')
        plt.tight_layout(); st.pyplot(f)
    with cc:
        f,ax = fig(5,3.5)
        sa = df.groupby('subject')['num_subscribers'].mean().sort_values()
        bars = ax.barh(sa.index, sa.values, color=PAL[:len(sa)], height=.5)
        ax.bar_label(bars, fmt='%.0f', fontsize=7, padding=3, color='#6b7280')
        ax.set_title("Inscrits moyens / Sujet", fontsize=10, fontweight='bold')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(f)

    st.markdown('<div class="sec">Top 10 Cours <span></span></div>', unsafe_allow_html=True)
    top10 = df.nlargest(10,'num_subscribers')[['course_title','subject','level','price','num_subscribers','num_reviews']].reset_index(drop=True)
    top10.index += 1; top10.columns = ['Titre','Sujet','Niveau','Prix ($)','Inscrits','Avis']
    st.dataframe(top10.style.background_gradient(subset=['Inscrits'], cmap='Blues'), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DSO 1
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  DSO 1 — Classification":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🎯 DSO 1 — Classification Binaire</div>
        <div class="page-sub">Prédire la popularité d'un cours · XGBoost · CRISP-DM</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#eef2ff;">🎯</div><div class="kpi-badge">Score</div></div><div class="kpi-val">~85%</div><div class="kpi-lbl">Accuracy</div></div>
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#ecfdf5;">⚡</div><div class="kpi-badge">Score</div></div><div class="kpi-val">~84%</div><div class="kpi-lbl">F1-Score</div></div>
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#fefce8;">📈</div><div class="kpi-badge">Score</div></div><div class="kpi-val">~0.92</div><div class="kpi-lbl">AUC-ROC</div></div>
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#f0f9ff;">📊</div><div class="kpi-badge">Seuil</div></div><div class="kpi-val">{threshold:,.0f}</div><div class="kpi-lbl">Inscrits médiane</div></div>
    </div>""", unsafe_allow_html=True)

    col_form, col_imp = st.columns([3,2], gap="large")
    with col_form:
        st.markdown('<div class="sec">Simulateur <span></span></div>', unsafe_allow_html=True)
        r1,r2 = st.columns(2)
        with r1:
            subject = st.selectbox("Sujet", df['subject'].unique())
            level   = st.selectbox("Niveau", df['level'].unique())
            is_paid = st.radio("Type", ["Payant","Gratuit"], horizontal=True)
        with r2:
            price    = st.slider("Prix ($)", 0, 200, 50) if is_paid=="Payant" else 0
            num_lec  = st.slider("Leçons", 0, 200, 30)
            duration = st.slider("Durée (h)", 0.0, 40.0, 5.0, .5)
            reviews  = st.number_input("Reviews estimées", 0, 5000, 100)

        if st.button("Analyser ce cours →", use_container_width=True):
            rps = reviews/100 if reviews>0 else 0
            lph = num_lec/duration if duration>0 else 0
            inp = {'is_paid':[1 if is_paid=="Payant" else 0],'price':[price],
                   'num_reviews':[reviews],'num_lectures':[num_lec],
                   'content_duration':[duration],'reviews_per_sub':[rps],'lectures_per_hour':[lph]}
            for s in ['Graphic Design','Musical Instruments','Web Development']:
                k = f'subject_{s}'
                if k in feat_cols: inp[k] = [1 if subject==s else 0]
            for lv in ['Expert Level','Intermediate Level']:
                k = f'level_{lv}'
                if k in feat_cols: inp[k] = [1 if level==lv else 0]
            Xin  = pd.DataFrame(inp).reindex(columns=feat_cols, fill_value=0)
            pred = model.predict(Xin)[0]
            proba= model.predict_proba(Xin)[0]
            if pred==1:
                st.markdown(f"""
                <div class="res-yes">
                    <div style="font-size:.72rem;color:#16a34a;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem;">✓ Populaire</div>
                    <div class="res-title">🏆 Ce cours sera populaire</div>
                    <div style="font-size:.83rem;color:#6b7280;margin-top:.3rem;">Dépasse probablement {threshold:,.0f} inscrits</div>
                    <div class="res-score" style="color:#16a34a;">{proba[1]*100:.1f}%</div>
                    <div style="font-size:.73rem;color:#9ca3af;">confiance</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="res-no">
                    <div style="font-size:.72rem;color:#d97706;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem;">✗ Peu populaire</div>
                    <div class="res-title">📉 Cours peu populaire</div>
                    <div style="font-size:.83rem;color:#6b7280;margin-top:.3rem;">Risque de ne pas atteindre {threshold:,.0f} inscrits</div>
                    <div class="res-score" style="color:#d97706;">{proba[0]*100:.1f}%</div>
                    <div style="font-size:.73rem;color:#9ca3af;">confiance</div>
                </div>""", unsafe_allow_html=True)
            f2,ax2 = fig(6,.7)
            ax2.barh([""], [proba[0]], color='#fde68a', height=.35, label='Non populaire')
            ax2.barh([""], [proba[1]], left=[proba[0]], color='#6ee7b7', height=.35, label='Populaire')
            ax2.set_xlim(0,1); ax2.set_yticks([])
            ax2.legend(loc='upper right', fontsize=8, framealpha=0)
            ax2.spines[['top','right','left','bottom']].set_visible(False)
            plt.tight_layout(); st.pyplot(f2)

    with col_imp:
        st.markdown('<div class="sec">Feature Importance <span></span></div>', unsafe_allow_html=True)
        imp = pd.DataFrame({'F':feat_cols,'S':model.feature_importances_}).sort_values('S',ascending=False).head(10)
        f2,ax2 = fig(5,5)
        ax2.barh(imp['F'][::-1], imp['S'][::-1], color=PAL[0], height=.55, alpha=.85)
        ax2.set_title("Top 10 Features — XGBoost", fontsize=10, fontweight='bold')
        ax2.spines[['top','right','left','bottom']].set_visible(False)
        ax2.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(f2)

        st.markdown('<div class="sec">Distribution Inscrits <span></span></div>', unsafe_allow_html=True)
        f2,ax2 = fig(5,3)
        ax2.hist(df['num_subscribers'].clip(upper=df['num_subscribers'].quantile(.95)),
                bins=40, color='#a5b4fc', edgecolor='white', linewidth=.5)
        ax2.axvline(threshold, color='#4f46e5', linewidth=2, linestyle='--', label=f'Seuil: {threshold:,.0f}')
        ax2.set_title("Distribution inscrits", fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8, framealpha=0)
        ax2.spines[['top','right','left','bottom']].set_visible(False)
        ax2.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(f2)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DSO 2
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵  DSO 2 — Clustering":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🔵 DSO 2 — Segmentation des Cours</div>
        <div class="page-sub">Clustering automatique · KMeans · Score de silhouette optimisé</div>
    </div>""", unsafe_allow_html=True)

    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    df2   = prepare_features(df)
    feats = ['price','num_reviews','num_lectures','content_duration','reviews_per_sub','lectures_per_hour']
    X     = df2[feats].fillna(0)
    Xsc   = scaler.transform(X)
    labels= kmeans.predict(Xsc)
    best_k= kmeans.n_clusters
    best_sil = silhouette_score(Xsc, labels)
    coords= PCA(n_components=2, random_state=42).fit_transform(Xsc)
    df2['cluster'] = labels
    csizes = pd.Series(labels).value_counts().sort_index()

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#eef2ff;">🔵</div><div class="kpi-badge">K</div></div><div class="kpi-val">{best_k}</div><div class="kpi-lbl">Clusters optimaux</div></div>
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#ecfdf5;">📐</div><div class="kpi-badge">Score</div></div><div class="kpi-val">{best_sil:.3f}</div><div class="kpi-lbl">Silhouette</div></div>
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#fefce8;">📦</div><div class="kpi-badge">Max</div></div><div class="kpi-val">{csizes.max()}</div><div class="kpi-lbl">Plus grand cluster</div></div>
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#f0f9ff;">📏</div><div class="kpi-badge">Min</div></div><div class="kpi-val">{csizes.min()}</div><div class="kpi-lbl">Plus petit cluster</div></div>
    </div>""", unsafe_allow_html=True)

    csc, csil = st.columns([3,2], gap="large")
    with csc:
        st.markdown('<div class="sec">Visualisation PCA 2D <span></span></div>', unsafe_allow_html=True)
        f2,ax2 = fig(7,5)
        for i in range(best_k):
            mask = labels==i
            ax2.scatter(coords[mask,0], coords[mask,1], c=PAL[i%len(PAL)],
                       s=20, alpha=.65, label=f"Cluster {i}", edgecolors='white', linewidths=.3)
        ax2.set_title(f"Segmentation en {best_k} clusters — PCA", fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8, framealpha=0)
        ax2.spines[['top','right','left','bottom']].set_visible(False)
        ax2.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(f2)

    with csil:
        st.markdown('<div class="sec">Taille des clusters <span></span></div>', unsafe_allow_html=True)
        f2,ax2 = fig(5,3)
        bars = ax2.bar([f"C{i}" for i in csizes.index], csizes.values,
                      color=PAL[:len(csizes)], width=.6, edgecolor='white', linewidth=1.5)
        ax2.bar_label(bars, fontsize=8, padding=3, color='#6b7280')
        ax2.spines[['top','right','left','bottom']].set_visible(False)
        ax2.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(f2)

    st.markdown('<div class="sec">Profil moyen par Cluster <span></span></div>', unsafe_allow_html=True)
    prof = df2[df2['cluster']>=0].groupby('cluster')[feats+['num_subscribers']].mean().round(2)
    prof['Nb cours'] = df2['cluster'].value_counts().sort_index()
    prof.index = [f"Cluster {i}" for i in prof.index]
    prof.columns = ['Prix','Avis','Leçons','Durée(h)','Avis/Sub','Leçons/h','Inscrits','Nb cours']
    st.dataframe(prof.style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown('<div class="sec">Sujets par Cluster <span></span></div>', unsafe_allow_html=True)
    df2m = df2.copy(); df2m['subject'] = df['subject'].values
    cross = pd.crosstab(df2m['cluster'], df2m['subject'])
    cross.index = [f"Cluster {i}" for i in cross.index]
    f2,ax2 = fig(12,3.5)
    cross.plot(kind='bar', ax=ax2, width=.7, color=PAL[:len(cross.columns)])
    ax2.set_title("Distribution sujets par cluster", fontsize=11, fontweight='bold')
    ax2.set_xlabel(""); ax2.tick_params(axis='x', rotation=0, labelsize=9)
    ax2.legend(fontsize=8, loc='upper right', framealpha=0)
    ax2.spines[['top','right','left','bottom']].set_visible(False)
    ax2.tick_params(left=False, bottom=False)
    plt.tight_layout(); st.pyplot(f2)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DSO 3
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚  DSO 3 — Recommandation":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📚 DSO 3 — Moteur de Recommandation</div>
        <div class="page-sub">Système basé sur le contenu · TF-IDF + KNN · Similarité cosinus</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#eef2ff;">📚</div><div class="kpi-badge">Total</div></div><div class="kpi-val">{len(df):,}</div><div class="kpi-lbl">Cours indexés</div></div>
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#ecfdf5;">🔤</div><div class="kpi-badge">TF-IDF</div></div><div class="kpi-val">5K</div><div class="kpi-lbl">Features</div></div>
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#fefce8;">📐</div><div class="kpi-badge">Métrique</div></div><div class="kpi-val">Cosine</div><div class="kpi-lbl">Similarité</div></div>
        <div class="kpi-card"><div class="kpi-top"><div class="kpi-icon" style="background:#f0f9ff;">⚡</div><div class="kpi-badge">Algo</div></div><div class="kpi-val">Brute</div><div class="kpi-lbl">KNN</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">Filtres & Sélection <span></span></div>', unsafe_allow_html=True)
    f1,f2,f3,f4 = st.columns([2,2,2,1])
    with f1: sf = st.selectbox("Sujet", ["Tous"]+sorted(df['subject'].unique().tolist()))
    with f2: lf = st.selectbox("Niveau", ["Tous"]+sorted(df['level'].unique().tolist()))
    with f3: tf = st.radio("Type", ["Tous","Payants","Gratuits"], horizontal=True)
    with f4: top_n = st.slider("Recs", 3, 10, 6)

    dff = df.copy()
    if sf!="Tous": dff = dff[dff['subject']==sf]
    if lf!="Tous": dff = dff[dff['level']==lf]
    if tf=="Payants": dff = dff[dff['price']>0]
    elif tf=="Gratuits": dff = dff[dff['price']==0]
    if dff.empty: st.warning("Aucun cours ne correspond aux filtres."); st.stop()

    selected = st.selectbox("Cours de référence", dff['course_title'].tolist())
    cb,_ = st.columns([1,4])
    with cb: run = st.button("Trouver des cours similaires →", use_container_width=True)

    if run:
        ref = df[df['course_title']==selected].iloc[0]
        paid_str = f"{int(ref['price'])}$" if ref['price']>0 else "Gratuit"
        st.markdown(f"""
        <div class="card" style="border-left:4px solid #4f46e5;">
            <div style="font-size:.68rem;color:#4f46e5;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem;">Cours de référence</div>
            <div style="font-size:1rem;font-weight:700;color:#111827;margin-bottom:.4rem;">{ref['course_title']}</div>
            <div style="font-size:.8rem;color:#9ca3af;">
                {ref['subject']} · {ref['level']} · {paid_str} · {int(ref['num_subscribers']):,} inscrits · {int(ref['num_reviews']):,} avis · {int(ref['num_lectures'])} leçons
            </div>
        </div>""", unsafe_allow_html=True)

        idx = df[df['course_title']==selected].index[0]
        dists,idxs = knn_model.kneighbors(matrix[idx], n_neighbors=top_n+1)
        recs = df.iloc[idxs[0][1:]].copy()
        recs['similarity'] = (1-dists[0][1:])
        recs = recs.sort_values('similarity', ascending=False).reset_index(drop=True)

        cl,cr = st.columns([3,2], gap="large")
        with cl:
            st.markdown('<div class="sec">Cours recommandés <span></span></div>', unsafe_allow_html=True)
            for i,row in recs.iterrows():
                pct = row['similarity']*100
                pr  = f"{int(row['price'])}$" if row['price']>0 else "Gratuit"
                st.markdown(f"""
                <div class="rec">
                    <div class="rec-n">#{i+1}</div>
                    <div style="flex:1;min-width:0;">
                        <div class="rec-t">{row['course_title']}</div>
                        <div class="rec-m">{row['subject']} · {row['level']} · {pr} · {int(row['num_subscribers']):,} inscrits</div>
                        <div class="pb"><div class="pf" style="width:{int(pct)}%"></div></div>
                    </div>
                    <div class="rec-p">{pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="sec">Scores de similarité <span></span></div>', unsafe_allow_html=True)
            f2,ax2 = fig(5,4)
            colors = ['#4f46e5' if s>=recs['similarity'].mean() else '#c7d2fe' for s in recs['similarity']]
            bars = ax2.bar(range(1,len(recs)+1), recs['similarity']*100,
                          color=colors, width=.6, edgecolor='white', linewidth=1.5)
            ax2.set_ylim(0,115)
            ax2.set_xticks(range(1,len(recs)+1))
            ax2.set_xticklabels([f"#{i+1}" for i in range(len(recs))], fontsize=8)
            ax2.bar_label(bars, fmt='%.0f%%', fontsize=7, padding=2, color='#6b7280')
            ax2.set_title("Similarité cosinus (%)", fontsize=10, fontweight='bold')
            ax2.spines[['top','right','left','bottom']].set_visible(False)
            ax2.tick_params(left=False, bottom=False)
            plt.tight_layout(); st.pyplot(f2)
