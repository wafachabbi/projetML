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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #f8faff;
    color: #1e293b;
}
.stApp { background: #f8faff; }
.block-container { padding: 2rem 2.5rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f1f5ff 100%);
    border-right: 1px solid #e2e8f8;
    box-shadow: 4px 0 20px rgba(99,102,241,.06);
}
[data-testid="stSidebar"] * { color: #475569 !important; }
[data-testid="stSidebar"] hr { border-color: #e2e8f8; }
[data-testid="stSidebar"] .stRadio label {
    background: #f8faff !important;
    border: 1px solid #e2e8f8 !important;
    border-radius: 10px !important;
    padding: .5rem 1rem !important;
    margin: .2rem 0 !important;
    transition: all .2s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #ede9fe !important;
    border-color: #a5b4fc !important;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(99,102,241,.3);
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: rgba(255,255,255,.08);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -50px; left: 25%;
    width: 220px; height: 220px;
    background: rgba(255,255,255,.05);
    border-radius: 50%;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    background: rgba(255,255,255,.2);
    border: 1px solid rgba(255,255,255,.3);
    border-radius: 99px;
    padding: .3rem .9rem;
    font-size: .68rem; font-weight: 600;
    color: rgba(255,255,255,.9);
    letter-spacing: 1.5px; text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem; font-weight: 800;
    color: #ffffff;
    line-height: 1.1; margin-bottom: .8rem;
}
.hero-sub { font-size: .95rem; color: rgba(255,255,255,.75); font-weight: 400; line-height: 1.6; max-width: 580px; }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin: 1.5rem 0; }
.kpi-card {
    background: #ffffff;
    border: 1px solid #e8edf8;
    border-radius: 18px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(99,102,241,.06);
    transition: transform .2s, box-shadow .2s;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(99,102,241,.12); }
.kpi-card::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 3px; border-radius: 0 0 18px 18px;
}
.kpi-card.v1::after { background: linear-gradient(90deg,#6366f1,#8b5cf6); }
.kpi-card.v2::after { background: linear-gradient(90deg,#06b6d4,#0891b2); }
.kpi-card.v3::after { background: linear-gradient(90deg,#f59e0b,#f97316); }
.kpi-card.v4::after { background: linear-gradient(90deg,#10b981,#059669); }
.kpi-icon-wrap {
    width: 44px; height: 44px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; margin-bottom: 1rem;
}
.kpi-card.v1 .kpi-icon-wrap { background: #ede9fe; }
.kpi-card.v2 .kpi-icon-wrap { background: #cffafe; }
.kpi-card.v3 .kpi-icon-wrap { background: #fef3c7; }
.kpi-card.v4 .kpi-icon-wrap { background: #d1fae5; }
.kpi-val { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:800; color:#0f172a; line-height:1; }
.kpi-lbl { font-size:.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1.5px; margin-top:.4rem; }
.kpi-trend { font-size:.78rem; color:#6366f1; margin-top:.5rem; font-weight:600; }

/* ── Section header ── */
.sec-head {
    display:flex; align-items:center; gap:.8rem;
    font-family:'Space Grotesk',sans-serif;
    font-size:1rem; font-weight:700; color:#0f172a;
    margin: 2.5rem 0 1.2rem 0;
}
.sec-dot { width:8px; height:8px; border-radius:50%; background:linear-gradient(135deg,#6366f1,#8b5cf6); flex-shrink:0; }
.sec-line { flex:1; height:1px; background:linear-gradient(90deg,#e2e8f8,transparent); }

/* ── White cards ── */
.w-card {
    background:#ffffff; border:1px solid #e8edf8;
    border-radius:18px; padding:1.5rem;
    box-shadow:0 2px 16px rgba(99,102,241,.05);
    margin-bottom:1rem;
}

/* ── DSO cards ── */
.dso-card {
    background:#ffffff; border:1px solid #e8edf8;
    border-radius:18px; padding:1.8rem;
    box-shadow:0 2px 16px rgba(99,102,241,.05);
    position:relative; overflow:hidden;
    transition: transform .2s, box-shadow .2s;
    height:100%;
}
.dso-card:hover { transform:translateY(-3px); box-shadow:0 8px 30px rgba(99,102,241,.12); }
.dso-num {
    font-family:'Space Grotesk',sans-serif; font-size:4rem; font-weight:800;
    position:absolute; top:.5rem; right:1.2rem;
    opacity:.05; color:#6366f1;
}
.dso-icon { font-size:2rem; margin-bottom:.8rem; }
.dso-title { font-family:'Space Grotesk',sans-serif; font-size:.95rem; font-weight:700; color:#0f172a; margin-bottom:.6rem; }
.dso-desc { font-size:.82rem; color:#64748b; line-height:1.7; }
.dso-tag {
    display:inline-block;
    background:#ede9fe; border-radius:99px;
    padding:.25rem .75rem; font-size:.68rem;
    color:#6366f1; margin-top:.8rem; font-weight:600;
}

/* ── Prediction ── */
.pred-popular {
    background:linear-gradient(135deg,#f0fdf4,#dcfce7);
    border:1.5px solid #86efac; border-radius:18px; padding:1.8rem; margin:1rem 0;
}
.pred-unpopular {
    background:linear-gradient(135deg,#fff7ed,#ffedd5);
    border:1.5px solid #fdba74; border-radius:18px; padding:1.8rem; margin:1rem 0;
}
.pred-title { font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:800; color:#0f172a; }
.pred-score { font-family:'Space Grotesk',sans-serif; font-size:2.5rem; font-weight:800; margin-top:.5rem; }

/* ── Rec cards ── */
.rec-item {
    background:#ffffff; border:1px solid #e8edf8;
    border-radius:14px; padding:1rem 1.2rem; margin:.5rem 0;
    display:flex; align-items:center; gap:1rem;
    box-shadow:0 1px 8px rgba(99,102,241,.04);
    transition: border-color .2s, box-shadow .2s;
}
.rec-item:hover { border-color:#a5b4fc; box-shadow:0 4px 16px rgba(99,102,241,.1); }
.rec-rank { font-family:'Space Grotesk',sans-serif; font-size:1.3rem; font-weight:800; color:#e0e7ff; min-width:2.5rem; }
.rec-title { font-weight:600; font-size:.9rem; color:#0f172a; }
.rec-meta { font-size:.75rem; color:#94a3b8; margin-top:.2rem; }
.rec-score { font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:800; color:#6366f1; white-space:nowrap; }
.bar-bg { background:#f1f5f9; border-radius:99px; height:3px; margin-top:.4rem; }
.bar-fg { background:linear-gradient(90deg,#6366f1,#8b5cf6); height:3px; border-radius:99px; }

/* ── Buttons ── */
.stButton > button {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color:white !important; border:none !important;
    border-radius:12px !important; font-weight:600 !important;
    padding:.6rem 1.5rem !important; transition:opacity .2s !important;
    box-shadow:0 4px 15px rgba(99,102,241,.3) !important;
}
.stButton > button:hover { opacity:.88 !important; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

GRAD = ['#6366f1','#8b5cf6','#06b6d4','#10b981','#f59e0b','#ec4899','#ef4444']

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

def light_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8faff')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor('#e2e8f8')
    ax.title.set_color('#0f172a')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    return fig, ax

df = load_data()
model, feat_cols, threshold, kmeans, scaler, tfidf, knn_model, matrix = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1.2rem;text-align:center;'>
        <div style='font-size:2.2rem;margin-bottom:.4rem;'>🧠</div>
        <div style='font-family:Space Grotesk,sans-serif;font-size:1.4rem;font-weight:800;
                    background:linear-gradient(135deg,#6366f1,#8b5cf6);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Smartek AI</div>
        <div style='font-size:.6rem;color:#cbd5e1;letter-spacing:3px;text-transform:uppercase;margin-top:.2rem;'>Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:.62rem;color:#cbd5e1;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.8rem;padding-left:.3rem;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", ["🏠  Vue Générale","🎯  DSO 1 — Classification","🔵  DSO 2 — Clustering","📚  DSO 3 — Recommandation"])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.72rem;color:#94a3b8;line-height:2.4;padding-left:.3rem;'>
        <div style='color:#6366f1;font-weight:700;font-size:.75rem;margin-bottom:.2rem;'>📦 Dataset</div>
        Udemy Courses · 3 678 cours<br>4 sujets · 4 niveaux<br><br>
        <div style='color:#6366f1;font-weight:700;font-size:.75rem;margin-bottom:.2rem;'>🤖 Modèles</div>
        XGBoost · KMeans<br>TF-IDF + KNN Cosine<br><br>
        <div style='color:#6366f1;font-weight:700;font-size:.75rem;margin-bottom:.2rem;'>📐 Méthode</div>
        CRISP-DM
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — VUE GÉNÉRALE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Vue Générale":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">✦ CRISP-DM · Machine Learning · 2025</div>
        <div class="hero-title">Smartek AI Dashboard</div>
        <div class="hero-sub">Plateforme d'analyse intelligente des cours Udemy — Classification, Clustering & Recommandation par IA.</div>
    </div>""", unsafe_allow_html=True)

    total=len(df); avg_sub=int(df['num_subscribers'].mean())
    free_pct=round((df['price']==0).mean()*100,1); avg_rev=int(df['num_reviews'].mean())
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card v1"><div class="kpi-icon-wrap">🎓</div><div class="kpi-val">{total:,}</div><div class="kpi-lbl">Cours indexés</div><div class="kpi-trend">↑ catalogue complet</div></div>
        <div class="kpi-card v2"><div class="kpi-icon-wrap">👥</div><div class="kpi-val">{avg_sub:,}</div><div class="kpi-lbl">Inscrits moyens</div><div class="kpi-trend">par cours</div></div>
        <div class="kpi-card v3"><div class="kpi-icon-wrap">⭐</div><div class="kpi-val">{avg_rev:,}</div><div class="kpi-lbl">Avis moyens</div><div class="kpi-trend">par cours</div></div>
        <div class="kpi-card v4"><div class="kpi-icon-wrap">🆓</div><div class="kpi-val">{free_pct}%</div><div class="kpi-lbl">Cours gratuits</div><div class="kpi-trend">du catalogue</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><div class="sec-dot"></div>Objectifs Data Science<div class="sec-line"></div></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3, gap="medium")
    for col,icon,num,title,desc,tag in [
        (c1,"🎯","01","DSO 1 — Classification","Prédire si un cours sera populaire ou non à partir de ses caractéristiques.","XGBoost"),
        (c2,"🔵","02","DSO 2 — Clustering","Segmenter les cours en groupes homogènes via KMeans optimisé.","KMeans + Silhouette"),
        (c3,"📚","03","DSO 3 — Recommandation","Suggérer des formations similaires via TF-IDF + similarité cosinus.","TF-IDF + KNN"),
    ]:
        with col:
            st.markdown(f"""
            <div class="dso-card">
                <div class="dso-num">{num}</div>
                <div class="dso-icon">{icon}</div>
                <div class="dso-title">{title}</div>
                <div class="dso-desc">{desc}</div>
                <div class="dso-tag">{tag}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><div class="sec-dot"></div>Aperçu du Dataset<div class="sec-line"></div></div>', unsafe_allow_html=True)
    ca,cb,cc = st.columns(3, gap="medium")
    with ca:
        fig,ax = light_fig(5,3.5)
        sc = df['subject'].value_counts()
        bars = ax.barh(sc.index, sc.values, color=GRAD[:len(sc)], height=.5)
        ax.bar_label(bars, fontsize=8, padding=4, color='#64748b')
        ax.set_title("Cours par Sujet", fontsize=10, fontweight='bold')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(fig)
    with cb:
        fig,ax = light_fig(5,3.5)
        lv = df['level'].value_counts()
        wedges,texts,autotexts = ax.pie(
            lv.values, labels=lv.index, colors=GRAD[:len(lv)],
            autopct='%1.0f%%', startangle=90,
            textprops={'fontsize':8,'color':'#475569'},
            wedgeprops={'edgecolor':'white','linewidth':2.5})
        for at in autotexts: at.set_fontweight('bold'); at.set_color('#0f172a')
        ax.set_title("Répartition par Niveau", fontsize=10, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig)
    with cc:
        fig,ax = light_fig(5,3.5)
        sa = df.groupby('subject')['num_subscribers'].mean().sort_values()
        bars = ax.barh(sa.index, sa.values, color=GRAD[:len(sa)], height=.5)
        ax.bar_label(bars, fmt='%.0f', fontsize=7, padding=4, color='#64748b')
        ax.set_title("Inscrits moyens / Sujet", fontsize=10, fontweight='bold')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="sec-head"><div class="sec-dot"></div>Top 10 Cours<div class="sec-line"></div></div>', unsafe_allow_html=True)
    top10 = df.nlargest(10,'num_subscribers')[['course_title','subject','level','price','num_subscribers','num_reviews']].reset_index(drop=True)
    top10.index += 1; top10.columns = ['Titre','Sujet','Niveau','Prix ($)','Inscrits','Avis']
    st.dataframe(top10.style.background_gradient(subset=['Inscrits'], cmap='Purples'), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DSO 1
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  DSO 1 — Classification":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">✦ DSO 1 · XGBoost · Classification Binaire</div>
        <div class="hero-title">Prédiction de Popularité</div>
        <div class="hero-sub">Déterminez si un cours Udemy atteindra le seuil de popularité grâce à notre modèle XGBoost.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card v1"><div class="kpi-icon-wrap">🎯</div><div class="kpi-val">~85%</div><div class="kpi-lbl">Accuracy</div><div class="kpi-trend">test set</div></div>
        <div class="kpi-card v2"><div class="kpi-icon-wrap">⚡</div><div class="kpi-val">~84%</div><div class="kpi-lbl">F1-Score</div><div class="kpi-trend">prec/recall</div></div>
        <div class="kpi-card v3"><div class="kpi-icon-wrap">📈</div><div class="kpi-val">~0.92</div><div class="kpi-lbl">AUC-ROC</div><div class="kpi-trend">discriminant</div></div>
        <div class="kpi-card v4"><div class="kpi-icon-wrap">📊</div><div class="kpi-val">{threshold:,.0f}</div><div class="kpi-lbl">Seuil popularité</div><div class="kpi-trend">inscrits médiane</div></div>
    </div>""", unsafe_allow_html=True)

    col_form, col_imp = st.columns([3,2], gap="large")
    with col_form:
        st.markdown('<div class="sec-head"><div class="sec-dot"></div>Simulateur de cours<div class="sec-line"></div></div>', unsafe_allow_html=True)
        r1,r2 = st.columns(2)
        with r1:
            subject = st.selectbox("📂 Sujet", df['subject'].unique())
            level   = st.selectbox("📊 Niveau", df['level'].unique())
            is_paid = st.radio("💳 Type", ["Payant","Gratuit"], horizontal=True)
        with r2:
            price    = st.slider("💰 Prix ($)", 0, 200, 50) if is_paid=="Payant" else 0
            num_lec  = st.slider("🎬 Leçons", 0, 200, 30)
            duration = st.slider("⏱ Durée (h)", 0.0, 40.0, 5.0, .5)
            reviews  = st.number_input("⭐ Reviews estimées", 0, 5000, 100)

        if st.button("🚀  Analyser ce cours", use_container_width=True):
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
                <div class="pred-popular">
                    <div style="font-size:.7rem;color:#16a34a;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:.5rem;">✓ Résultat</div>
                    <div class="pred-title">🏆 Cours Populaire</div>
                    <div style="font-size:.85rem;color:#64748b;margin-top:.3rem;">Dépasse probablement {threshold:,.0f} inscrits</div>
                    <div class="pred-score" style="color:#16a34a;">{proba[1]*100:.1f}%</div>
                    <div style="font-size:.75rem;color:#94a3b8;">indice de confiance</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-unpopular">
                    <div style="font-size:.7rem;color:#ea580c;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:.5rem;">✗ Résultat</div>
                    <div class="pred-title">📉 Cours Peu Populaire</div>
                    <div style="font-size:.85rem;color:#64748b;margin-top:.3rem;">Risque de ne pas atteindre {threshold:,.0f} inscrits</div>
                    <div class="pred-score" style="color:#ea580c;">{proba[0]*100:.1f}%</div>
                    <div style="font-size:.75rem;color:#94a3b8;">indice de confiance</div>
                </div>""", unsafe_allow_html=True)
            fig,ax = light_fig(6,.8)
            ax.barh([""], [proba[0]], color='#fca5a5', height=.4, label='Non populaire')
            ax.barh([""], [proba[1]], left=[proba[0]], color='#86efac', height=.4, label='Populaire')
            ax.set_xlim(0,1); ax.set_yticks([])
            ax.legend(loc='upper right', fontsize=8, framealpha=0)
            ax.spines[['top','right','left','bottom']].set_visible(False)
            plt.tight_layout(); st.pyplot(fig)

    with col_imp:
        st.markdown('<div class="sec-head"><div class="sec-dot"></div>Feature Importance<div class="sec-line"></div></div>', unsafe_allow_html=True)
        imp = pd.DataFrame({'F':feat_cols,'S':model.feature_importances_}).sort_values('S',ascending=False).head(10)
        fig,ax = light_fig(5,5)
        colors = [GRAD[min(i,len(GRAD)-1)] for i in range(len(imp))]
        ax.barh(imp['F'][::-1], imp['S'][::-1], color=colors[::-1], height=.55)
        ax.set_title("Top 10 Features — XGBoost", fontsize=10, fontweight='bold')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(fig)

        st.markdown('<div class="sec-head"><div class="sec-dot"></div>Distribution Inscrits<div class="sec-line"></div></div>', unsafe_allow_html=True)
        fig,ax = light_fig(5,3)
        ax.hist(df['num_subscribers'].clip(upper=df['num_subscribers'].quantile(.95)),
                bins=40, color='#a5b4fc', edgecolor='white', linewidth=.5)
        ax.axvline(threshold, color='#6366f1', linewidth=2, linestyle='--', label=f'Seuil: {threshold:,.0f}')
        ax.set_title("Distribution inscrits", fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, framealpha=0)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DSO 2
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵  DSO 2 — Clustering":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">✦ DSO 2 · KMeans · Segmentation</div>
        <div class="hero-title">Segmentation des Cours</div>
        <div class="hero-sub">Regroupement automatique des cours en segments homogènes via KMeans optimisé par le score de silhouette.</div>
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
        <div class="kpi-card v1"><div class="kpi-icon-wrap">🔵</div><div class="kpi-val">{best_k}</div><div class="kpi-lbl">Clusters optimaux</div><div class="kpi-trend">score silhouette</div></div>
        <div class="kpi-card v2"><div class="kpi-icon-wrap">📐</div><div class="kpi-val">{best_sil:.3f}</div><div class="kpi-lbl">Score Silhouette</div><div class="kpi-trend">K={best_k}</div></div>
        <div class="kpi-card v3"><div class="kpi-icon-wrap">📦</div><div class="kpi-val">{csizes.max()}</div><div class="kpi-lbl">Plus grand cluster</div><div class="kpi-trend">nb cours</div></div>
        <div class="kpi-card v4"><div class="kpi-icon-wrap">📏</div><div class="kpi-val">{csizes.min()}</div><div class="kpi-lbl">Plus petit cluster</div><div class="kpi-trend">nb cours</div></div>
    </div>""", unsafe_allow_html=True)

    csc, csil = st.columns([3,2], gap="large")
    with csc:
        st.markdown('<div class="sec-head"><div class="sec-dot"></div>Visualisation PCA 2D<div class="sec-line"></div></div>', unsafe_allow_html=True)
        fig,ax = light_fig(7,5)
        for i in range(best_k):
            mask = labels==i
            ax.scatter(coords[mask,0], coords[mask,1], c=GRAD[i%len(GRAD)],
                       s=22, alpha=.65, label=f"Cluster {i}", edgecolors='white', linewidths=.3)
        ax.set_title(f"Segmentation en {best_k} clusters — PCA", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, framealpha=0)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(fig)

    with csil:
        st.markdown('<div class="sec-head"><div class="sec-dot"></div>Taille des clusters<div class="sec-line"></div></div>', unsafe_allow_html=True)
        fig,ax = light_fig(5,3)
        bars = ax.bar([f"C{i}" for i in csizes.index], csizes.values,
                      color=GRAD[:len(csizes)], width=.6, edgecolor='white', linewidth=1.5)
        ax.bar_label(bars, fontsize=8, padding=3, color='#64748b')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="sec-head"><div class="sec-dot"></div>Profil moyen par Cluster<div class="sec-line"></div></div>', unsafe_allow_html=True)
    prof = df2[df2['cluster']>=0].groupby('cluster')[feats+['num_subscribers']].mean().round(2)
    prof['Nb cours'] = df2['cluster'].value_counts().sort_index()
    prof.index = [f"Cluster {i}" for i in prof.index]
    prof.columns = ['Prix','Avis','Leçons','Durée(h)','Avis/Sub','Leçons/h','Inscrits','Nb cours']
    st.dataframe(prof.style.background_gradient(cmap='Purples'), use_container_width=True)

    st.markdown('<div class="sec-head"><div class="sec-dot"></div>Sujets par Cluster<div class="sec-line"></div></div>', unsafe_allow_html=True)
    df2m = df2.copy(); df2m['subject'] = df['subject'].values
    cross = pd.crosstab(df2m['cluster'], df2m['subject'])
    cross.index = [f"Cluster {i}" for i in cross.index]
    fig,ax = light_fig(12,3.5)
    cross.plot(kind='bar', ax=ax, width=.7, color=GRAD[:len(cross.columns)])
    ax.set_title("Distribution sujets par cluster", fontsize=11, fontweight='bold')
    ax.set_xlabel(""); ax.tick_params(axis='x', rotation=0, labelsize=9)
    ax.legend(fontsize=8, loc='upper right', framealpha=0)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DSO 3
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚  DSO 3 — Recommandation":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">✦ DSO 3 · TF-IDF · Similarité Cosinus</div>
        <div class="hero-title">Moteur de Recommandation</div>
        <div class="hero-sub">Découvrez des cours similaires grâce à notre moteur TF-IDF + KNN basé sur la similarité cosinus.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card v1"><div class="kpi-icon-wrap">📚</div><div class="kpi-val">{len(df):,}</div><div class="kpi-lbl">Cours indexés</div><div class="kpi-trend">moteur actif</div></div>
        <div class="kpi-card v2"><div class="kpi-icon-wrap">🔤</div><div class="kpi-val">5K</div><div class="kpi-lbl">Features TF-IDF</div><div class="kpi-trend">n-grammes 1-2</div></div>
        <div class="kpi-card v3"><div class="kpi-icon-wrap">📐</div><div class="kpi-val">Cosine</div><div class="kpi-lbl">Similarité</div><div class="kpi-trend">métrique KNN</div></div>
        <div class="kpi-card v4"><div class="kpi-icon-wrap">⚡</div><div class="kpi-val">Brute</div><div class="kpi-lbl">Algorithme</div><div class="kpi-trend">recherche exacte</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><div class="sec-dot"></div>Filtres & Sélection<div class="sec-line"></div></div>', unsafe_allow_html=True)
    f1,f2,f3,f4 = st.columns([2,2,2,1])
    with f1: sf = st.selectbox("📂 Sujet", ["Tous"]+sorted(df['subject'].unique().tolist()))
    with f2: lf = st.selectbox("📊 Niveau", ["Tous"]+sorted(df['level'].unique().tolist()))
    with f3: tf = st.radio("💳 Type", ["Tous","Payants","Gratuits"], horizontal=True)
    with f4: top_n = st.slider("🔢 Recs", 3, 10, 6)

    dff = df.copy()
    if sf!="Tous": dff = dff[dff['subject']==sf]
    if lf!="Tous": dff = dff[dff['level']==lf]
    if tf=="Payants": dff = dff[dff['price']>0]
    elif tf=="Gratuits": dff = dff[dff['price']==0]
    if dff.empty: st.warning("Aucun cours ne correspond aux filtres."); st.stop()

    selected = st.selectbox("📌 Cours de référence", dff['course_title'].tolist())
    cb,_ = st.columns([1,4])
    with cb: run = st.button("🚀  Trouver des cours similaires", use_container_width=True)

    if run:
        ref = df[df['course_title']==selected].iloc[0]
        paid_str = f"{int(ref['price'])}$" if ref['price']>0 else "Gratuit"
        st.markdown(f"""
        <div class="w-card" style="border-left:4px solid #6366f1;margin-bottom:1.5rem;">
            <div style="font-size:.65rem;color:#6366f1;text-transform:uppercase;letter-spacing:2px;font-weight:700;margin-bottom:.5rem;">Cours de référence</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:.5rem;">{ref['course_title']}</div>
            <div style="font-size:.8rem;color:#94a3b8;">
                📂 {ref['subject']} &nbsp;·&nbsp; 📊 {ref['level']} &nbsp;·&nbsp; 💰 {paid_str}
                &nbsp;·&nbsp; 👥 {int(ref['num_subscribers']):,} inscrits
                &nbsp;·&nbsp; ⭐ {int(ref['num_reviews']):,} avis
                &nbsp;·&nbsp; 🎬 {int(ref['num_lectures'])} leçons
            </div>
        </div>""", unsafe_allow_html=True)

        idx = df[df['course_title']==selected].index[0]
        dists,idxs = knn_model.kneighbors(matrix[idx], n_neighbors=top_n+1)
        recs = df.iloc[idxs[0][1:]].copy()
        recs['similarity'] = (1-dists[0][1:])
        recs = recs.sort_values('similarity', ascending=False).reset_index(drop=True)

        cl,cr = st.columns([3,2], gap="large")
        with cl:
            st.markdown('<div class="sec-head"><div class="sec-dot"></div>Cours recommandés<div class="sec-line"></div></div>', unsafe_allow_html=True)
            for i,row in recs.iterrows():
                pct = row['similarity']*100
                pr  = f"{int(row['price'])}$" if row['price']>0 else "Gratuit"
                st.markdown(f"""
                <div class="rec-item">
                    <div class="rec-rank">#{i+1}</div>
                    <div style="flex:1;min-width:0;">
                        <div class="rec-title">{row['course_title']}</div>
                        <div class="rec-meta">📂 {row['subject']} · 📊 {row['level']} · 💰 {pr} · 👥 {int(row['num_subscribers']):,}</div>
                        <div class="bar-bg"><div class="bar-fg" style="width:{int(pct)}%"></div></div>
                    </div>
                    <div class="rec-score">{pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="sec-head"><div class="sec-dot"></div>Scores de similarité<div class="sec-line"></div></div>', unsafe_allow_html=True)
            fig,ax = light_fig(5,4)
            colors = ['#6366f1' if s>=recs['similarity'].mean() else '#e0e7ff' for s in recs['similarity']]
            bars = ax.bar(range(1,len(recs)+1), recs['similarity']*100,
                          color=colors, width=.6, edgecolor='white', linewidth=1.5)
            ax.set_ylim(0,115)
            ax.set_xticks(range(1,len(recs)+1))
            ax.set_xticklabels([f"#{i+1}" for i in range(len(recs))], fontsize=8)
            ax.bar_label(bars, fmt='%.0f%%', fontsize=7, padding=2, color='#64748b')
            ax.set_title("Similarité cosinus (%)", fontsize=10, fontweight='bold')
            ax.spines[['top','right','left','bottom']].set_visible(False)
            ax.tick_params(left=False, bottom=False)
            plt.tight_layout(); st.pyplot(fig)
