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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Nunito:wght@600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #f0f7ff;
    color: #1a2e4a;
}
.stApp { background: #f0f7ff; }
.block-container { padding: 1.8rem 2.5rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0ea5e9 0%, #0284c7 40%, #0369a1 100%);
    border-right: none;
    box-shadow: 4px 0 24px rgba(14,165,233,.25);
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.15); }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #ffffff !important; }
[data-testid="stSidebar"] .stRadio > div { gap: .3rem !important; }
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,.12) !important;
    border: 1px solid rgba(255,255,255,.25) !important;
    border-radius: 12px !important;
    padding: .55rem 1rem !important;
    margin: .15rem 0 !important;
    transition: all .2s !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,.25) !important;
    border-color: rgba(255,255,255,.5) !important;
}
[data-testid="stSidebar"] .stRadio input[type="radio"] { accent-color: #ffffff; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(120deg, #0ea5e9 0%, #38bdf8 45%, #22d3ee 100%);
    border-radius: 22px;
    padding: 2.8rem 3rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(14,165,233,.28);
}
.hero::before {
    content: '';
    position: absolute; top: -70px; right: -70px;
    width: 280px; height: 280px;
    background: rgba(255,255,255,.12); border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute; bottom: -50px; left: 30%;
    width: 200px; height: 200px;
    background: rgba(255,255,255,.07); border-radius: 50%;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    background: rgba(255,255,255,.25);
    border: 1px solid rgba(255,255,255,.4);
    border-radius: 99px; padding: .28rem .9rem;
    font-size: .68rem; font-weight: 700;
    color: #fff; letter-spacing: 1.8px; text-transform: uppercase;
    margin-bottom: .9rem;
}
.hero-title {
    font-family: 'Nunito', sans-serif;
    font-size: 2.6rem; font-weight: 900;
    color: #ffffff; line-height: 1.1; margin-bottom: .7rem;
}
.hero-sub { font-size: .92rem; color: rgba(255,255,255,.82); line-height: 1.65; max-width: 560px; }

/* ── KPI grid ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin: 1.5rem 0; }
.kpi-card {
    background: #ffffff;
    border-radius: 18px; padding: 1.4rem 1.5rem;
    border: 1.5px solid #e0f2fe;
    box-shadow: 0 2px 14px rgba(14,165,233,.07);
    transition: transform .2s, box-shadow .2s;
    position: relative; overflow: hidden;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(14,165,233,.15); }
.kpi-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 4px; border-radius: 18px 18px 0 0;
}
.kpi-card.b::before { background: linear-gradient(90deg,#0ea5e9,#38bdf8); }
.kpi-card.g::before { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-card.t::before { background: linear-gradient(90deg,#06b6d4,#22d3ee); }
.kpi-card.a::before { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.kpi-icon {
    width: 46px; height: 46px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; margin-bottom: .9rem;
}
.kpi-card.b .kpi-icon { background: #e0f2fe; }
.kpi-card.g .kpi-icon { background: #d1fae5; }
.kpi-card.t .kpi-icon { background: #cffafe; }
.kpi-card.a .kpi-icon { background: #dbeafe; }
.kpi-val { font-family:'Nunito',sans-serif; font-size:2rem; font-weight:900; color:#0f172a; line-height:1; }
.kpi-lbl { font-size:.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1.4px; margin-top:.35rem; }
.kpi-sub { font-size:.76rem; color:#0ea5e9; margin-top:.4rem; font-weight:600; }

/* ── Section title ── */
.stitle {
    font-family:'Nunito',sans-serif; font-size:1.05rem; font-weight:800;
    color:#0f172a; margin: 2rem 0 1rem 0;
    display:flex; align-items:center; gap:.7rem;
}
.stitle-bar { width:4px; height:20px; border-radius:99px; background:linear-gradient(180deg,#0ea5e9,#10b981); flex-shrink:0; }
.stitle-line { flex:1; height:1px; background:linear-gradient(90deg,#bae6fd,transparent); }

/* ── Feature cards ── */
.feat-card {
    background:#ffffff; border-radius:18px; padding:1.6rem;
    border:1.5px solid #e0f2fe;
    box-shadow:0 2px 14px rgba(14,165,233,.06);
    transition: transform .2s, box-shadow .2s;
    position:relative; overflow:hidden; height:100%;
}
.feat-card:hover { transform:translateY(-3px); box-shadow:0 8px 28px rgba(14,165,233,.13); }
.feat-card-num {
    font-family:'Nunito',sans-serif; font-size:4.5rem; font-weight:900;
    position:absolute; top:.2rem; right:1rem;
    opacity:.04; color:#0ea5e9; line-height:1;
}
.feat-icon { font-size:2rem; margin-bottom:.7rem; }
.feat-title { font-family:'Nunito',sans-serif; font-size:.95rem; font-weight:800; color:#0f172a; margin-bottom:.5rem; }
.feat-desc { font-size:.82rem; color:#64748b; line-height:1.7; }
.feat-tag {
    display:inline-block; background:#e0f2fe;
    border-radius:99px; padding:.22rem .75rem;
    font-size:.68rem; color:#0284c7; margin-top:.8rem; font-weight:700;
}

/* ── Prediction boxes ── */
.box-yes {
    background:linear-gradient(135deg,#f0fdf4,#dcfce7);
    border:2px solid #86efac; border-radius:18px; padding:1.8rem; margin:1rem 0;
}
.box-no {
    background:linear-gradient(135deg,#fff7ed,#ffedd5);
    border:2px solid #fcd34d; border-radius:18px; padding:1.8rem; margin:1rem 0;
}
.box-title { font-family:'Nunito',sans-serif; font-size:1.4rem; font-weight:900; color:#0f172a; }
.box-score { font-family:'Nunito',sans-serif; font-size:2.6rem; font-weight:900; margin-top:.4rem; }

/* ── Rec items ── */
.rec-row {
    background:#ffffff; border:1.5px solid #e0f2fe;
    border-radius:14px; padding:1rem 1.2rem; margin:.45rem 0;
    display:flex; align-items:center; gap:1rem;
    box-shadow:0 1px 8px rgba(14,165,233,.05);
    transition: border-color .2s, box-shadow .2s;
}
.rec-row:hover { border-color:#7dd3fc; box-shadow:0 4px 16px rgba(14,165,233,.12); }
.rec-num { font-family:'Nunito',sans-serif; font-size:1.3rem; font-weight:900; color:#bae6fd; min-width:2.2rem; }
.rec-name { font-weight:600; font-size:.9rem; color:#0f172a; }
.rec-info { font-size:.74rem; color:#94a3b8; margin-top:.18rem; }
.rec-pct { font-family:'Nunito',sans-serif; font-size:1rem; font-weight:800; color:#0ea5e9; white-space:nowrap; }
.prog-bg { background:#e0f2fe; border-radius:99px; height:3px; margin-top:.35rem; }
.prog-fg { background:linear-gradient(90deg,#0ea5e9,#10b981); height:3px; border-radius:99px; }

/* ── Buttons ── */
.stButton > button {
    background:linear-gradient(135deg,#0ea5e9,#10b981) !important;
    color:#fff !important; border:none !important;
    border-radius:12px !important; font-weight:700 !important;
    padding:.6rem 1.5rem !important;
    box-shadow:0 4px 15px rgba(14,165,233,.3) !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity:.88 !important; }
footer { visibility:hidden; }
/* Force sidebar open */
[data-testid="stSidebar"] { min-width: 260px !important; max-width: 260px !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] > div { width: 260px !important; }
</style>
""", unsafe_allow_html=True)

COLORS = ['#0ea5e9','#10b981','#06b6d4','#3b82f6','#f59e0b','#8b5cf6','#ec4899']

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

def mfig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f0f7ff')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#e0f2fe')
    ax.title.set_color('#0f172a')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    return fig, ax

df = load_data()
model, feat_cols, threshold, kmeans, scaler, tfidf, knn_model, matrix = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.6rem 0 1.2rem;text-align:center;'>
        <div style='font-size:2.4rem;margin-bottom:.4rem;'>🧠</div>
        <div style='font-family:Nunito,sans-serif;font-size:1.5rem;font-weight:900;color:#fff;'>Smartek AI</div>
        <div style='font-size:.6rem;color:rgba(255,255,255,.5);letter-spacing:3px;text-transform:uppercase;margin-top:.2rem;'>Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:.62rem;color:rgba(255,255,255,.4);letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.8rem;padding-left:.3rem;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", ["🏠  Vue Générale","🎯  DSO 1 — Classification","🔵  DSO 2 — Clustering","📚  DSO 3 — Recommandation"])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.73rem;color:rgba(255,255,255,.55);line-height:2.3;padding-left:.3rem;'>
        <div style='color:#7dd3fc;font-weight:800;font-size:.75rem;margin-bottom:.2rem;'>📦 Dataset</div>
        Udemy Courses · 3 678 cours<br>4 sujets · 4 niveaux<br><br>
        <div style='color:#7dd3fc;font-weight:800;font-size:.75rem;margin-bottom:.2rem;'>🤖 Modèles</div>
        XGBoost · KMeans<br>TF-IDF + KNN Cosine<br><br>
        <div style='color:#7dd3fc;font-weight:800;font-size:.75rem;margin-bottom:.2rem;'>📐 Méthode</div>
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
        <div class="kpi-card b"><div class="kpi-icon">🎓</div><div class="kpi-val">{total:,}</div><div class="kpi-lbl">Cours indexés</div><div class="kpi-sub">↑ catalogue complet</div></div>
        <div class="kpi-card g"><div class="kpi-icon">👥</div><div class="kpi-val">{avg_sub:,}</div><div class="kpi-lbl">Inscrits moyens</div><div class="kpi-sub">par cours</div></div>
        <div class="kpi-card t"><div class="kpi-icon">⭐</div><div class="kpi-val">{avg_rev:,}</div><div class="kpi-lbl">Avis moyens</div><div class="kpi-sub">par cours</div></div>
        <div class="kpi-card a"><div class="kpi-icon">🆓</div><div class="kpi-val">{free_pct}%</div><div class="kpi-lbl">Cours gratuits</div><div class="kpi-sub">du catalogue</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="stitle"><div class="stitle-bar"></div>Objectifs Data Science<div class="stitle-line"></div></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3, gap="medium")
    for col,icon,num,title,desc,tag in [
        (c1,"🎯","01","DSO 1 — Classification","Prédire si un cours sera populaire ou non à partir de ses caractéristiques pour orienter la création de contenu.","XGBoost"),
        (c2,"🔵","02","DSO 2 — Clustering","Segmenter les cours en groupes homogènes via KMeans optimisé par le score de silhouette.","KMeans + Silhouette"),
        (c3,"📚","03","DSO 3 — Recommandation","Suggérer des formations similaires via TF-IDF + similarité cosinus pour augmenter les inscriptions.","TF-IDF + KNN"),
    ]:
        with col:
            st.markdown(f"""
            <div class="feat-card">
                <div class="feat-card-num">{num}</div>
                <div class="feat-icon">{icon}</div>
                <div class="feat-title">{title}</div>
                <div class="feat-desc">{desc}</div>
                <div class="feat-tag">{tag}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="stitle"><div class="stitle-bar"></div>Aperçu du Dataset<div class="stitle-line"></div></div>', unsafe_allow_html=True)
    ca,cb,cc = st.columns(3, gap="medium")
    with ca:
        fig,ax = mfig(5,3.5)
        sc = df['subject'].value_counts()
        bars = ax.barh(sc.index, sc.values, color=COLORS[:len(sc)], height=.5)
        ax.bar_label(bars, fontsize=8, padding=4, color='#64748b')
        ax.set_title("Cours par Sujet", fontsize=10, fontweight='bold')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(fig)
    with cb:
        fig,ax = mfig(5,3.5)
        lv = df['level'].value_counts()
        wedges,_,autotexts = ax.pie(
            lv.values, labels=lv.index, colors=COLORS[:len(lv)],
            autopct='%1.0f%%', startangle=90,
            textprops={'fontsize':8,'color':'#475569'},
            wedgeprops={'edgecolor':'white','linewidth':2.5})
        for at in autotexts: at.set_fontweight('bold'); at.set_color('#0f172a')
        ax.set_title("Répartition par Niveau", fontsize=10, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig)
    with cc:
        fig,ax = mfig(5,3.5)
        sa = df.groupby('subject')['num_subscribers'].mean().sort_values()
        bars = ax.barh(sa.index, sa.values, color=COLORS[:len(sa)], height=.5)
        ax.bar_label(bars, fmt='%.0f', fontsize=7, padding=4, color='#64748b')
        ax.set_title("Inscrits moyens / Sujet", fontsize=10, fontweight='bold')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="stitle"><div class="stitle-bar"></div>Top 10 Cours<div class="stitle-line"></div></div>', unsafe_allow_html=True)
    top10 = df.nlargest(10,'num_subscribers')[['course_title','subject','level','price','num_subscribers','num_reviews']].reset_index(drop=True)
    top10.index += 1; top10.columns = ['Titre','Sujet','Niveau','Prix ($)','Inscrits','Avis']
    st.dataframe(top10.style.background_gradient(subset=['Inscrits'], cmap='Blues'), use_container_width=True)

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
        <div class="kpi-card b"><div class="kpi-icon">🎯</div><div class="kpi-val">~85%</div><div class="kpi-lbl">Accuracy</div><div class="kpi-sub">test set</div></div>
        <div class="kpi-card g"><div class="kpi-icon">⚡</div><div class="kpi-val">~84%</div><div class="kpi-lbl">F1-Score</div><div class="kpi-sub">prec/recall</div></div>
        <div class="kpi-card t"><div class="kpi-icon">📈</div><div class="kpi-val">~0.92</div><div class="kpi-lbl">AUC-ROC</div><div class="kpi-sub">discriminant</div></div>
        <div class="kpi-card a"><div class="kpi-icon">📊</div><div class="kpi-val">{threshold:,.0f}</div><div class="kpi-lbl">Seuil popularité</div><div class="kpi-sub">inscrits médiane</div></div>
    </div>""", unsafe_allow_html=True)

    col_form, col_imp = st.columns([3,2], gap="large")
    with col_form:
        st.markdown('<div class="stitle"><div class="stitle-bar"></div>Simulateur de cours<div class="stitle-line"></div></div>', unsafe_allow_html=True)
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
                <div class="box-yes">
                    <div style="font-size:.7rem;color:#16a34a;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:.4rem;">✓ Résultat</div>
                    <div class="box-title">🏆 Cours Populaire</div>
                    <div style="font-size:.85rem;color:#64748b;margin-top:.3rem;">Dépasse probablement {threshold:,.0f} inscrits</div>
                    <div class="box-score" style="color:#16a34a;">{proba[1]*100:.1f}%</div>
                    <div style="font-size:.74rem;color:#94a3b8;">indice de confiance</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="box-no">
                    <div style="font-size:.7rem;color:#d97706;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:.4rem;">✗ Résultat</div>
                    <div class="box-title">📉 Cours Peu Populaire</div>
                    <div style="font-size:.85rem;color:#64748b;margin-top:.3rem;">Risque de ne pas atteindre {threshold:,.0f} inscrits</div>
                    <div class="box-score" style="color:#d97706;">{proba[0]*100:.1f}%</div>
                    <div style="font-size:.74rem;color:#94a3b8;">indice de confiance</div>
                </div>""", unsafe_allow_html=True)
            fig,ax = mfig(6,.8)
            ax.barh([""], [proba[0]], color='#fcd34d', height=.4, label='Non populaire')
            ax.barh([""], [proba[1]], left=[proba[0]], color='#34d399', height=.4, label='Populaire')
            ax.set_xlim(0,1); ax.set_yticks([])
            ax.legend(loc='upper right', fontsize=8, framealpha=0)
            ax.spines[['top','right','left','bottom']].set_visible(False)
            plt.tight_layout(); st.pyplot(fig)

    with col_imp:
        st.markdown('<div class="stitle"><div class="stitle-bar"></div>Feature Importance<div class="stitle-line"></div></div>', unsafe_allow_html=True)
        imp = pd.DataFrame({'F':feat_cols,'S':model.feature_importances_}).sort_values('S',ascending=False).head(10)
        fig,ax = mfig(5,5)
        ax.barh(imp['F'][::-1], imp['S'][::-1], color=COLORS[:len(imp)][::-1], height=.55)
        ax.set_title("Top 10 Features — XGBoost", fontsize=10, fontweight='bold')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(fig)

        st.markdown('<div class="stitle"><div class="stitle-bar"></div>Distribution Inscrits<div class="stitle-line"></div></div>', unsafe_allow_html=True)
        fig,ax = mfig(5,3)
        ax.hist(df['num_subscribers'].clip(upper=df['num_subscribers'].quantile(.95)),
                bins=40, color='#7dd3fc', edgecolor='white', linewidth=.5)
        ax.axvline(threshold, color='#0ea5e9', linewidth=2, linestyle='--', label=f'Seuil: {threshold:,.0f}')
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
        <div class="kpi-card b"><div class="kpi-icon">🔵</div><div class="kpi-val">{best_k}</div><div class="kpi-lbl">Clusters optimaux</div><div class="kpi-sub">score silhouette</div></div>
        <div class="kpi-card g"><div class="kpi-icon">📐</div><div class="kpi-val">{best_sil:.3f}</div><div class="kpi-lbl">Score Silhouette</div><div class="kpi-sub">K={best_k}</div></div>
        <div class="kpi-card t"><div class="kpi-icon">📦</div><div class="kpi-val">{csizes.max()}</div><div class="kpi-lbl">Plus grand cluster</div><div class="kpi-sub">nb cours</div></div>
        <div class="kpi-card a"><div class="kpi-icon">📏</div><div class="kpi-val">{csizes.min()}</div><div class="kpi-lbl">Plus petit cluster</div><div class="kpi-sub">nb cours</div></div>
    </div>""", unsafe_allow_html=True)

    csc, csil = st.columns([3,2], gap="large")
    with csc:
        st.markdown('<div class="stitle"><div class="stitle-bar"></div>Visualisation PCA 2D<div class="stitle-line"></div></div>', unsafe_allow_html=True)
        fig,ax = mfig(7,5)
        for i in range(best_k):
            mask = labels==i
            ax.scatter(coords[mask,0], coords[mask,1], c=COLORS[i%len(COLORS)],
                       s=22, alpha=.65, label=f"Cluster {i}", edgecolors='white', linewidths=.3)
        ax.set_title(f"Segmentation en {best_k} clusters — PCA", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, framealpha=0)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(fig)

    with csil:
        st.markdown('<div class="stitle"><div class="stitle-bar"></div>Taille des clusters<div class="stitle-line"></div></div>', unsafe_allow_html=True)
        fig,ax = mfig(5,3)
        bars = ax.bar([f"C{i}" for i in csizes.index], csizes.values,
                      color=COLORS[:len(csizes)], width=.6, edgecolor='white', linewidth=1.5)
        ax.bar_label(bars, fontsize=8, padding=3, color='#64748b')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="stitle"><div class="stitle-bar"></div>Profil moyen par Cluster<div class="stitle-line"></div></div>', unsafe_allow_html=True)
    prof = df2[df2['cluster']>=0].groupby('cluster')[feats+['num_subscribers']].mean().round(2)
    prof['Nb cours'] = df2['cluster'].value_counts().sort_index()
    prof.index = [f"Cluster {i}" for i in prof.index]
    prof.columns = ['Prix','Avis','Leçons','Durée(h)','Avis/Sub','Leçons/h','Inscrits','Nb cours']
    st.dataframe(prof.style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown('<div class="stitle"><div class="stitle-bar"></div>Sujets par Cluster<div class="stitle-line"></div></div>', unsafe_allow_html=True)
    df2m = df2.copy(); df2m['subject'] = df['subject'].values
    cross = pd.crosstab(df2m['cluster'], df2m['subject'])
    cross.index = [f"Cluster {i}" for i in cross.index]
    fig,ax = mfig(12,3.5)
    cross.plot(kind='bar', ax=ax, width=.7, color=COLORS[:len(cross.columns)])
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
        <div class="kpi-card b"><div class="kpi-icon">📚</div><div class="kpi-val">{len(df):,}</div><div class="kpi-lbl">Cours indexés</div><div class="kpi-sub">moteur actif</div></div>
        <div class="kpi-card g"><div class="kpi-icon">🔤</div><div class="kpi-val">5K</div><div class="kpi-lbl">Features TF-IDF</div><div class="kpi-sub">n-grammes 1-2</div></div>
        <div class="kpi-card t"><div class="kpi-icon">📐</div><div class="kpi-val">Cosine</div><div class="kpi-lbl">Similarité</div><div class="kpi-sub">métrique KNN</div></div>
        <div class="kpi-card a"><div class="kpi-icon">⚡</div><div class="kpi-val">Brute</div><div class="kpi-lbl">Algorithme</div><div class="kpi-sub">recherche exacte</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="stitle"><div class="stitle-bar"></div>Filtres & Sélection<div class="stitle-line"></div></div>', unsafe_allow_html=True)
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
        <div style="background:#ffffff;border:1.5px solid #bae6fd;border-left:5px solid #0ea5e9;
                    border-radius:16px;padding:1.3rem 1.5rem;margin-bottom:1.5rem;
                    box-shadow:0 2px 14px rgba(14,165,233,.08);">
            <div style="font-size:.65rem;color:#0ea5e9;text-transform:uppercase;letter-spacing:2px;font-weight:700;margin-bottom:.4rem;">Cours de référence</div>
            <div style="font-family:'Nunito',sans-serif;font-size:1.1rem;font-weight:800;color:#0f172a;margin-bottom:.4rem;">{ref['course_title']}</div>
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
            st.markdown('<div class="stitle"><div class="stitle-bar"></div>Cours recommandés<div class="stitle-line"></div></div>', unsafe_allow_html=True)
            for i,row in recs.iterrows():
                pct = row['similarity']*100
                pr  = f"{int(row['price'])}$" if row['price']>0 else "Gratuit"
                st.markdown(f"""
                <div class="rec-row">
                    <div class="rec-num">#{i+1}</div>
                    <div style="flex:1;min-width:0;">
                        <div class="rec-name">{row['course_title']}</div>
                        <div class="rec-info">📂 {row['subject']} · 📊 {row['level']} · 💰 {pr} · 👥 {int(row['num_subscribers']):,}</div>
                        <div class="prog-bg"><div class="prog-fg" style="width:{int(pct)}%"></div></div>
                    </div>
                    <div class="rec-pct">{pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="stitle"><div class="stitle-bar"></div>Scores de similarité<div class="stitle-line"></div></div>', unsafe_allow_html=True)
            fig,ax = mfig(5,4)
            colors = ['#0ea5e9' if s>=recs['similarity'].mean() else '#bae6fd' for s in recs['similarity']]
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
