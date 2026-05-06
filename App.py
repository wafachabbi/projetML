import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Smartek AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #f0f4ff; }
[data-testid="stSidebar"] { background: linear-gradient(160deg,#1a3a6e 0%,#0f2347 60%,#091830 100%); border-right:none; box-shadow:4px 0 24px rgba(15,35,71,.18); }
[data-testid="stSidebar"] * { color: #c8d8f8 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.1); }
.kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1.5rem 0; }
.kpi-card { background:white; border-radius:16px; padding:1.4rem 1.6rem; border:1px solid #e2e8f8; box-shadow:0 2px 12px rgba(15,35,71,.06); position:relative; overflow:hidden; }
.kpi-card::after { content:''; position:absolute; bottom:0;left:0;right:0; height:3px; border-radius:0 0 16px 16px; }
.kpi-card.c1::after { background:linear-gradient(90deg,#2563eb,#60a5fa); }
.kpi-card.c2::after { background:linear-gradient(90deg,#0891b2,#22d3ee); }
.kpi-card.c3::after { background:linear-gradient(90deg,#7c3aed,#a78bfa); }
.kpi-card.c4::after { background:linear-gradient(90deg,#0d9488,#34d399); }
.kpi-icon { font-size:1.5rem; margin-bottom:.5rem; }
.kpi-val  { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#0f2347; line-height:1; }
.kpi-lbl  { font-size:.72rem; color:#94a3c8; text-transform:uppercase; letter-spacing:1.2px; margin-top:.3rem; }
.kpi-sub  { font-size:.78rem; color:#2563eb; margin-top:.35rem; font-weight:600; }
.sec-head { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:800; color:#0f2347; letter-spacing:-.5px; margin:2rem 0 1rem 0; display:flex; align-items:center; gap:.6rem; }
.sec-line { flex:1; height:1px; background:linear-gradient(90deg,#e2e8f8,transparent); }
.white-card { background:white; border-radius:16px; padding:1.4rem 1.6rem; border:1px solid #e2e8f8; box-shadow:0 2px 12px rgba(15,35,71,.06); margin-bottom:1rem; }
.result-popular { background:linear-gradient(135deg,#eff6ff,#dbeafe); border:1.5px solid #93c5fd; border-radius:14px; padding:1.4rem 1.6rem; margin:1rem 0; }
.result-unpopular { background:linear-gradient(135deg,#fff7ed,#ffedd5); border:1.5px solid #fed7aa; border-radius:14px; padding:1.4rem 1.6rem; margin:1rem 0; }
.result-title { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; color:#0f2347; }
.result-sub { font-size:.85rem; margin-top:.3rem; color:#6b7eb8; }
.result-score { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; margin-top:.8rem; }
.rec-card { background:white; border:1px solid #e2e8f8; border-radius:12px; padding:.9rem 1.2rem; margin:.5rem 0; display:flex; align-items:center; gap:1rem; box-shadow:0 1px 6px rgba(15,35,71,.04); }
.rec-num  { font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:800; color:#dbeafe; min-width:2rem; }
.rec-body { flex:1; min-width:0; }
.rec-t    { font-weight:600; font-size:.9rem; color:#0f2347; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.rec-m    { font-size:.75rem; color:#94a3c8; margin-top:.2rem; }
.rec-pct  { font-family:'Syne',sans-serif; font-size:.95rem; font-weight:800; color:#2563eb; }
.sim-bg   { background:#f0f4ff; border-radius:99px; height:4px; margin-top:.4rem; }
.sim-fg   { background:linear-gradient(90deg,#2563eb,#60a5fa); height:4px; border-radius:99px; }
.smartek-title { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#0f2347; letter-spacing:-1.5px; line-height:1.1; }
.smartek-sub   { font-size:.9rem; color:#6b7eb8; margin-top:.3rem; font-weight:400; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

CLUSTER_COLORS = ['#2563eb','#0891b2','#7c3aed','#0d9488','#ea580c','#db2777','#ca8a04']

@st.cache_data
def load_data():
    return pd.read_csv("udemy_courses.csv")

@st.cache_data
def prepare_features(df):
    df2 = df.copy()
    for col in ['price','num_subscribers','num_reviews','num_lectures','content_duration']:
        Q1,Q3 = df2[col].quantile(.25), df2[col].quantile(.75)
        df2[col] = df2[col].clip(Q1-1.5*(Q3-Q1), Q3+1.5*(Q3-Q1))
    df2['reviews_per_sub']   = (df2['num_reviews']/df2['num_subscribers'].replace(0,np.nan)).fillna(0)
    df2['lectures_per_hour'] = (df2['num_lectures']/df2['content_duration'].replace(0,np.nan)).fillna(0)
    return df2

@st.cache_resource
def train_classifier(_df):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from xgboost import XGBClassifier
    df2 = prepare_features(_df)
    threshold = df2['num_subscribers'].median()
    y = (df2['num_subscribers'] >= threshold).astype(int)
    drop = ['course_id','url','published_timestamp','course_title','num_subscribers']
    X = df2.drop(columns=[c for c in drop if c in df2.columns])
    X = pd.get_dummies(X, columns=['subject','level'], drop_first=True)
    X['is_paid'] = X['is_paid'].astype(int)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42)
    mdl = XGBClassifier(n_estimators=200,max_depth=5,learning_rate=.1,subsample=.8,
                        colsample_bytree=.8,eval_metric='logloss',random_state=42,verbosity=0)
    mdl.fit(Xtr,ytr)
    yp = mdl.predict(Xte); ypr = mdl.predict_proba(Xte)[:,1]
    metrics = {'accuracy':accuracy_score(yte,yp),'f1':f1_score(yte,yp),'auc':roc_auc_score(yte,ypr)}
    return mdl, X.columns.tolist(), threshold, metrics

@st.cache_resource
def train_clustering(_df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    df2 = prepare_features(_df)
    feats = ['price','num_reviews','num_lectures','content_duration','reviews_per_sub','lectures_per_hour']
    X = df2[feats].fillna(0)
    Xsc = StandardScaler().fit_transform(X)
    best_k,best_sil,sil_scores = 4,-1,{}
    for k in range(2,9):
        km = KMeans(n_clusters=k,random_state=42,n_init=10)
        s  = silhouette_score(Xsc, km.fit_predict(Xsc))
        sil_scores[k] = s
        if s > best_sil: best_sil,best_k = s,k
    km_f   = KMeans(n_clusters=best_k,random_state=42,n_init=10)
    labels = km_f.fit_predict(Xsc)
    coords = PCA(n_components=2,random_state=42).fit_transform(Xsc)
    return labels, coords, best_k, best_sil, sil_scores, df2, feats

@st.cache_resource
def train_recommender(_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    df2 = _df.copy()
    df2['combined'] = df2['course_title'].fillna('')+" "+df2['subject'].fillna('')+" "+df2['level'].fillna('')
    tfidf  = TfidfVectorizer(stop_words='english',max_features=5000,ngram_range=(1,2))
    matrix = tfidf.fit_transform(df2['combined'])
    knn    = NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=11)
    knn.fit(matrix)
    return tfidf, matrix, knn

df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem 0;text-align:center;'>
        <div style='font-family:Syne,sans-serif;font-size:1.7rem;font-weight:800;color:#60a5fa;'>🧠 Smartek</div>
        <div style='font-size:.65rem;color:#4b6899;letter-spacing:2.5px;text-transform:uppercase;margin-top:.2rem;'>AI Dashboard</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:.68rem;color:#4b6899;letter-spacing:2px;text-transform:uppercase;margin-bottom:.7rem;padding-left:.2rem;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", ["🏠  Vue Générale","🎯  DSO 1 — Classification","🔵  DSO 2 — Clustering","📚  DSO 3 — Recommandation"])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.72rem;color:#4b6899;line-height:2.2;padding-left:.2rem;'>
        <span style='color:#60a5fa;font-weight:700;'>Dataset</span><br>Udemy Courses · 3 678 cours<br>4 sujets · 4 niveaux<br><br>
        <span style='color:#60a5fa;font-weight:700;'>Modèles</span><br>XGBoost · KMeans<br>TF-IDF + KNN Cosine<br><br>
        <span style='color:#60a5fa;font-weight:700;'>Méthode</span><br>CRISP-DM
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — VUE GÉNÉRALE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Vue Générale":
    st.markdown('<div class="smartek-title">Smartek AI Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="smartek-sub">Plateforme d\'analyse intelligente des cours · Projet CRISP-DM</div>', unsafe_allow_html=True)

    total=len(df); avg_sub=int(df['num_subscribers'].mean())
    free_pct=round((df['price']==0).mean()*100,1); avg_rev=int(df['num_reviews'].mean())
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card c1"><div class="kpi-icon">🎓</div><div class="kpi-val">{total:,}</div><div class="kpi-lbl">Cours indexés</div><div class="kpi-sub">↑ catalogue complet</div></div>
        <div class="kpi-card c2"><div class="kpi-icon">👥</div><div class="kpi-val">{avg_sub:,}</div><div class="kpi-lbl">Inscrits moyens</div><div class="kpi-sub">par cours</div></div>
        <div class="kpi-card c3"><div class="kpi-icon">⭐</div><div class="kpi-val">{avg_rev:,}</div><div class="kpi-lbl">Avis moyens</div><div class="kpi-sub">par cours</div></div>
        <div class="kpi-card c4"><div class="kpi-icon">🆓</div><div class="kpi-val">{free_pct}%</div><div class="kpi-lbl">Cours gratuits</div><div class="kpi-sub">du catalogue</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">📌 Objectifs Data Science <div class="sec-line"></div></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    dso_data = [
        (c1,"🎯","DSO 1 — Classification","Prédire si un cours sera populaire ou non à partir de ses caractéristiques pour orienter la création de contenu et l'allocation marketing."),
        (c2,"🔵","DSO 2 — Clustering","Segmenter les cours en groupes homogènes via KMeans optimisé par le score de silhouette pour améliorer la navigation du catalogue."),
        (c3,"📚","DSO 3 — Recommandation","Suggérer des formations similaires via TF-IDF + similarité cosinus pour proposer des contenus pertinents et augmenter les inscriptions."),
    ]
    for col,icon,label,desc in dso_data:
        with col:
            st.markdown(f"""
            <div class="white-card" style="border-top:3px solid #2563eb;">
                <div style="font-size:1.8rem;margin-bottom:.5rem;">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:.9rem;color:#0f2347;margin-bottom:.5rem;">{label}</div>
                <div style="font-size:.82rem;color:#6b7eb8;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">📊 Aperçu du Dataset <div class="sec-line"></div></div>', unsafe_allow_html=True)
    ca,cb,cc = st.columns(3)
    with ca:
        fig,ax=plt.subplots(figsize=(5,3.5))
        sc=df['subject'].value_counts()
        bars=ax.barh(sc.index,sc.values,color=['#2563eb','#0891b2','#7c3aed','#0d9488'],height=.55)
        ax.bar_label(bars,fontsize=8,padding=3)
        ax.set_title("Cours par Sujet",fontsize=10,fontweight='bold',color='#0f2347')
        ax.spines[['top','right']].set_visible(False); ax.tick_params(labelsize=8)
        fig.patch.set_facecolor('none'); ax.set_facecolor('#f8faff')
        plt.tight_layout(); st.pyplot(fig)
    with cb:
        fig,ax=plt.subplots(figsize=(5,3.5))
        lv=df['level'].value_counts()
        ax.pie(lv.values,labels=lv.index,colors=['#2563eb','#0891b2','#7c3aed','#0d9488'],
               autopct='%1.0f%%',startangle=90,textprops={'fontsize':8},wedgeprops={'edgecolor':'white','linewidth':2})
        ax.set_title("Répartition par Niveau",fontsize=10,fontweight='bold',color='#0f2347')
        fig.patch.set_facecolor('none'); plt.tight_layout(); st.pyplot(fig)
    with cc:
        fig,ax=plt.subplots(figsize=(5,3.5))
        sa=df.groupby('subject')['num_subscribers'].mean().sort_values()
        bars=ax.barh(sa.index,sa.values,color=['#0d9488','#7c3aed','#0891b2','#2563eb'],height=.55)
        ax.bar_label(bars,fmt='%.0f',fontsize=7,padding=3)
        ax.set_title("Inscrits moyens / Sujet",fontsize=10,fontweight='bold',color='#0f2347')
        ax.spines[['top','right']].set_visible(False); ax.tick_params(labelsize=8)
        fig.patch.set_facecolor('none'); ax.set_facecolor('#f8faff')
        plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="sec-head">🏆 Top 10 Cours <div class="sec-line"></div></div>', unsafe_allow_html=True)
    top10=df.nlargest(10,'num_subscribers')[['course_title','subject','level','price','num_subscribers','num_reviews']].reset_index(drop=True)
    top10.index+=1; top10.columns=['Titre','Sujet','Niveau','Prix ($)','Inscrits','Avis']
    st.dataframe(top10.style.background_gradient(subset=['Inscrits'],cmap='Blues'),use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DSO 1
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  DSO 1 — Classification":
    st.markdown('<div class="smartek-title">DSO 1 — Classification Binaire</div>', unsafe_allow_html=True)
    st.markdown('<div class="smartek-sub">Prédire la popularité d\'un cours · XGBoost · CRISP-DM</div>', unsafe_allow_html=True)

    with st.spinner("Entraînement XGBoost..."):
        model,feat_cols,threshold,metrics = train_classifier(df)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card c1"><div class="kpi-icon">🎯</div><div class="kpi-val">{metrics['accuracy']*100:.1f}%</div><div class="kpi-lbl">Accuracy</div><div class="kpi-sub">test set</div></div>
        <div class="kpi-card c2"><div class="kpi-icon">⚡</div><div class="kpi-val">{metrics['f1']*100:.1f}%</div><div class="kpi-lbl">F1-Score</div><div class="kpi-sub">prec/recall</div></div>
        <div class="kpi-card c3"><div class="kpi-icon">📈</div><div class="kpi-val">{metrics['auc']:.3f}</div><div class="kpi-lbl">AUC-ROC</div><div class="kpi-sub">discriminant</div></div>
        <div class="kpi-card c4"><div class="kpi-icon">📊</div><div class="kpi-val">{threshold:,.0f}</div><div class="kpi-lbl">Seuil popularité</div><div class="kpi-sub">inscrits médiane</div></div>
    </div>""", unsafe_allow_html=True)

    col_form,col_imp = st.columns([3,2],gap="large")

    with col_form:
        st.markdown('<div class="sec-head">🔧 Prédire un nouveau cours <div class="sec-line"></div></div>', unsafe_allow_html=True)
        r1,r2 = st.columns(2)
        with r1:
            subject = st.selectbox("📂 Sujet",df['subject'].unique())
            level   = st.selectbox("📊 Niveau",df['level'].unique())
            is_paid = st.radio("💳 Type",["Payant","Gratuit"],horizontal=True)
        with r2:
            price    = st.slider("💰 Prix ($)",0,200,50) if is_paid=="Payant" else 0
            num_lec  = st.slider("🎬 Leçons",0,200,30)
            duration = st.slider("⏱ Durée (h)",0.0,40.0,5.0,.5)
            reviews  = st.number_input("⭐ Reviews estimées",0,5000,100)

        if st.button("🚀  Prédire la popularité",use_container_width=True):
            rps = reviews/100 if reviews>0 else 0
            lph = num_lec/duration if duration>0 else 0
            inp = {'is_paid':[1 if is_paid=="Payant" else 0],'price':[price],'num_reviews':[reviews],
                   'num_lectures':[num_lec],'content_duration':[duration],'reviews_per_sub':[rps],'lectures_per_hour':[lph]}
            for s in ['Graphic Design','Musical Instruments','Web Development']:
                k=f'subject_{s}'
                if k in feat_cols: inp[k]=[1 if subject==s else 0]
            for lv in ['Expert Level','Intermediate Level']:
                k=f'level_{lv}'
                if k in feat_cols: inp[k]=[1 if level==lv else 0]
            Xin = pd.DataFrame(inp).reindex(columns=feat_cols,fill_value=0)
            pred=model.predict(Xin)[0]; proba=model.predict_proba(Xin)[0]
            if pred==1:
                st.markdown(f'<div class="result-popular"><div class="result-title">🏆 Cours Populaire</div><div class="result-sub">Dépasse probablement {threshold:,.0f} inscrits</div><div class="result-score" style="color:#2563eb;">Confiance : {proba[1]*100:.1f}%</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-unpopular"><div class="result-title">📉 Cours Peu Populaire</div><div class="result-sub">Risque de ne pas atteindre {threshold:,.0f} inscrits</div><div class="result-score" style="color:#ea580c;">Confiance : {proba[0]*100:.1f}%</div></div>', unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(6,.9))
            ax.barh([""], [proba[0]],color='#fed7aa',height=.5,label='Non populaire')
            ax.barh([""], [proba[1]],left=[proba[0]],color='#93c5fd',height=.5,label='Populaire')
            ax.set_xlim(0,1); ax.set_yticks([]); ax.legend(loc='upper right',fontsize=8)
            ax.spines[['top','right','left']].set_visible(False)
            fig.patch.set_facecolor('none'); ax.set_facecolor('none'); st.pyplot(fig)

    with col_imp:
        st.markdown('<div class="sec-head">📌 Feature Importance <div class="sec-line"></div></div>', unsafe_allow_html=True)
        imp=pd.DataFrame({'F':feat_cols,'S':model.feature_importances_}).sort_values('S',ascending=False).head(10)
        fig,ax=plt.subplots(figsize=(5,5))
        clrs=['#2563eb' if i==0 else '#93c5fd' for i in range(len(imp))]
        ax.barh(imp['F'][::-1],imp['S'][::-1],color=clrs[::-1],height=.6)
        ax.set_title("Top 10 Features — XGBoost",fontsize=10,fontweight='bold',color='#0f2347')
        ax.spines[['top','right']].set_visible(False); ax.tick_params(labelsize=8)
        fig.patch.set_facecolor('none'); ax.set_facecolor('#f8faff')
        plt.tight_layout(); st.pyplot(fig)

        st.markdown('<div class="sec-head">📊 Distribution Inscrits <div class="sec-line"></div></div>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(5,3))
        ax.hist(df['num_subscribers'].clip(upper=df['num_subscribers'].quantile(.95)),bins=40,color='#93c5fd',edgecolor='white',linewidth=.5)
        ax.axvline(threshold,color='#2563eb',linewidth=2,linestyle='--',label=f'Seuil:{threshold:,.0f}')
        ax.set_title("Distribution inscrits",fontsize=10,fontweight='bold',color='#0f2347')
        ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
        fig.patch.set_facecolor('none'); ax.set_facecolor('#f8faff')
        plt.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DSO 2
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵  DSO 2 — Clustering":
    st.markdown('<div class="smartek-title">DSO 2 — Segmentation des Cours</div>', unsafe_allow_html=True)
    st.markdown('<div class="smartek-sub">Clustering automatique · KMeans · Score de silhouette optimisé</div>', unsafe_allow_html=True)

    with st.spinner("Calcul des clusters..."):
        labels,coords,best_k,best_sil,sil_scores,df2,feats = train_clustering(df)

    df2['cluster']=labels; csizes=pd.Series(labels).value_counts().sort_index()
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card c1"><div class="kpi-icon">🔵</div><div class="kpi-val">{best_k}</div><div class="kpi-lbl">Clusters optimaux</div><div class="kpi-sub">score silhouette</div></div>
        <div class="kpi-card c2"><div class="kpi-icon">📐</div><div class="kpi-val">{best_sil:.3f}</div><div class="kpi-lbl">Score Silhouette</div><div class="kpi-sub">K={best_k}</div></div>
        <div class="kpi-card c3"><div class="kpi-icon">📦</div><div class="kpi-val">{csizes.max()}</div><div class="kpi-lbl">Plus grand cluster</div><div class="kpi-sub">nb cours</div></div>
        <div class="kpi-card c4"><div class="kpi-icon">📏</div><div class="kpi-val">{csizes.min()}</div><div class="kpi-lbl">Plus petit cluster</div><div class="kpi-sub">nb cours</div></div>
    </div>""", unsafe_allow_html=True)

    csc,csil=st.columns([3,2],gap="large")
    with csc:
        st.markdown('<div class="sec-head">🗺️ Visualisation PCA 2D <div class="sec-line"></div></div>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(7,5))
        for i in range(best_k):
            mask=labels==i
            ax.scatter(coords[mask,0],coords[mask,1],c=CLUSTER_COLORS[i%len(CLUSTER_COLORS)],
                       s=25,alpha=.65,label=f"Cluster {i}",edgecolors='white',linewidths=.3)
        ax.set_title(f"Segmentation en {best_k} clusters — PCA",fontsize=11,fontweight='bold',color='#0f2347')
        ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel("Composante 1",fontsize=9); ax.set_ylabel("Composante 2",fontsize=9)
        fig.patch.set_facecolor('none'); ax.set_facecolor('#f8faff')
        plt.tight_layout(); st.pyplot(fig)
    with csil:
        st.markdown('<div class="sec-head">📈 Score Silhouette / K <div class="sec-line"></div></div>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(5,3.5))
        ks=list(sil_scores.keys()); vs=list(sil_scores.values())
        bc=['#2563eb' if k==best_k else '#bfdbfe' for k in ks]
        bars=ax.bar(ks,vs,color=bc,width=.6,edgecolor='white',linewidth=1)
        ax.bar_label(bars,fmt='%.3f',fontsize=7.5,padding=3)
        ax.set_xlabel("K",fontsize=9); ax.set_ylabel("Silhouette",fontsize=9)
        ax.set_title("Optimisation du K",fontsize=10,fontweight='bold',color='#0f2347')
        ax.set_xticks(ks); ax.spines[['top','right']].set_visible(False)
        fig.patch.set_facecolor('none'); ax.set_facecolor('#f8faff')
        plt.tight_layout(); st.pyplot(fig)

        st.markdown('<div class="sec-head">📦 Taille des clusters <div class="sec-line"></div></div>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(5,2.5))
        ax.bar([f"C{i}" for i in csizes.index],csizes.values,
               color=CLUSTER_COLORS[:len(csizes)],width=.6,edgecolor='white',linewidth=1)
        ax.set_ylabel("Cours",fontsize=8); ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)
        fig.patch.set_facecolor('none'); ax.set_facecolor('#f8faff')
        plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="sec-head">📋 Profil moyen par Cluster <div class="sec-line"></div></div>', unsafe_allow_html=True)
    prof=df2[df2['cluster']>=0].groupby('cluster')[feats+['num_subscribers']].mean().round(2)
    prof['Nb cours']=df2['cluster'].value_counts().sort_index()
    prof.index=[f"Cluster {i}" for i in prof.index]
    prof.columns=['Prix','Avis','Leçons','Durée(h)','Avis/Sub','Leçons/h','Inscrits','Nb cours']
    st.dataframe(prof.style.background_gradient(cmap='Blues'),use_container_width=True)

    st.markdown('<div class="sec-head">📂 Sujets par Cluster <div class="sec-line"></div></div>', unsafe_allow_html=True)
    df2m=df2.copy(); df2m['subject']=df['subject'].values
    cross=pd.crosstab(df2m['cluster'],df2m['subject'])
    cross.index=[f"Cluster {i}" for i in cross.index]
    fig,ax=plt.subplots(figsize=(12,3.5))
    cross.plot(kind='bar',ax=ax,width=.7,color=['#2563eb','#0891b2','#7c3aed','#0d9488'])
    ax.set_title("Distribution sujets par cluster",fontsize=11,fontweight='bold',color='#0f2347')
    ax.set_xlabel(""); ax.tick_params(axis='x',rotation=0,labelsize=9)
    ax.legend(fontsize=8,loc='upper right'); ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('none'); ax.set_facecolor('#f8faff')
    plt.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DSO 3
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚  DSO 3 — Recommandation":
    st.markdown('<div class="smartek-title">DSO 3 — Recommandation de Cours</div>', unsafe_allow_html=True)
    st.markdown('<div class="smartek-sub">Système basé sur le contenu · TF-IDF + KNN · Similarité cosinus</div>', unsafe_allow_html=True)

    with st.spinner("Chargement du moteur..."):
        tfidf,matrix,knn_model = train_recommender(df)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card c1"><div class="kpi-icon">📚</div><div class="kpi-val">{len(df):,}</div><div class="kpi-lbl">Cours indexés</div><div class="kpi-sub">moteur actif</div></div>
        <div class="kpi-card c2"><div class="kpi-icon">🔤</div><div class="kpi-val">5K</div><div class="kpi-lbl">Features TF-IDF</div><div class="kpi-sub">n-grammes 1-2</div></div>
        <div class="kpi-card c3"><div class="kpi-icon">📐</div><div class="kpi-val">Cosine</div><div class="kpi-lbl">Similarité</div><div class="kpi-sub">métrique KNN</div></div>
        <div class="kpi-card c4"><div class="kpi-icon">⚡</div><div class="kpi-val">Brute</div><div class="kpi-lbl">Algorithme</div><div class="kpi-sub">recherche exacte</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">🔍 Filtres & Sélection <div class="sec-line"></div></div>', unsafe_allow_html=True)
    f1,f2,f3,f4=st.columns([2,2,2,1])
    with f1: sf=st.selectbox("📂 Sujet",["Tous"]+sorted(df['subject'].unique().tolist()))
    with f2: lf=st.selectbox("📊 Niveau",["Tous"]+sorted(df['level'].unique().tolist()))
    with f3: tf=st.radio("💳 Type",["Tous","Payants","Gratuits"],horizontal=True)
    with f4: top_n=st.slider("🔢 Recs",3,10,6)

    dff=df.copy()
    if sf!="Tous": dff=dff[dff['subject']==sf]
    if lf!="Tous": dff=dff[dff['level']==lf]
    if tf=="Payants": dff=dff[dff['price']>0]
    elif tf=="Gratuits": dff=dff[dff['price']==0]
    if dff.empty: st.warning("Aucun cours ne correspond aux filtres."); st.stop()

    selected=st.selectbox("📌 Cours de référence",dff['course_title'].tolist())
    cb,_=st.columns([1,4])
    with cb: run=st.button("🚀  Recommander",use_container_width=True)

    if run:
        ref=df[df['course_title']==selected].iloc[0]
        paid_str=f"{int(ref['price'])}$" if ref['price']>0 else "Gratuit"
        st.markdown(f"""
        <div class="white-card" style="border-left:4px solid #2563eb;">
            <div style="font-size:.7rem;color:#2563eb;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:.4rem;">Cours de référence</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#0f2347;margin-bottom:.4rem;">{ref['course_title']}</div>
            <div style="font-size:.82rem;color:#6b7eb8;">
                📂 {ref['subject']} &nbsp;·&nbsp; 📊 {ref['level']} &nbsp;·&nbsp; 💰 {paid_str}
                &nbsp;·&nbsp; 👥 {int(ref['num_subscribers']):,} inscrits
                &nbsp;·&nbsp; ⭐ {int(ref['num_reviews']):,} avis
                &nbsp;·&nbsp; 🎬 {int(ref['num_lectures'])} leçons &nbsp;·&nbsp; ⏱ {ref['content_duration']}h
            </div>
        </div>""", unsafe_allow_html=True)

        idx=df[df['course_title']==selected].index[0]
        dists,idxs=knn_model.kneighbors(matrix[idx],n_neighbors=top_n+1)
        recs=df.iloc[idxs[0][1:]].copy()
        recs['similarity']=(1-dists[0][1:])
        recs=recs.sort_values('similarity',ascending=False).reset_index(drop=True)

        cl,cr=st.columns([3,2],gap="large")
        with cl:
            st.markdown('<div class="sec-head">📋 Cours recommandés <div class="sec-line"></div></div>', unsafe_allow_html=True)
            for i,row in recs.iterrows():
                pct=row['similarity']*100
                pr=f"{int(row['price'])}$" if row['price']>0 else "Gratuit"
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-num">#{i+1}</div>
                    <div class="rec-body">
                        <div class="rec-t">{row['course_title']}</div>
                        <div class="rec-m">📂 {row['subject']} &nbsp;·&nbsp; 📊 {row['level']} &nbsp;·&nbsp; 💰 {pr} &nbsp;·&nbsp; 👥 {int(row['num_subscribers']):,}</div>
                        <div class="sim-bg"><div class="sim-fg" style="width:{int(pct)}%"></div></div>
                    </div>
                    <div class="rec-pct">{pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="sec-head">📊 Analyse <div class="sec-line"></div></div>', unsafe_allow_html=True)
            fig,axes=plt.subplots(3,1,figsize=(5,9)); fig.patch.set_facecolor('none')

            ax=axes[0]
            bc=['#2563eb' if s>=recs['similarity'].mean() else '#bfdbfe' for s in recs['similarity']]
            bars=ax.bar(range(1,len(recs)+1),recs['similarity']*100,color=bc,width=.6,edgecolor='white')
            ax.set_ylim(0,115); ax.set_xticks(range(1,len(recs)+1))
            ax.set_xticklabels([f"#{i+1}" for i in range(len(recs))],fontsize=8)
            ax.bar_label(bars,fmt='%.0f%%',fontsize=7,padding=2)
            ax.set_title("Scores de similarité",fontsize=9,fontweight='bold',color='#0f2347')
            ax.spines[['top','right']].set_visible(False); ax.set_facecolor('none')

            ax2=axes[1]
            sc2=recs['subject'].value_counts()
            ax2.pie(sc2.values,labels=sc2.index,colors=CLUSTER_COLORS[:len(sc2)],
                    autopct='%1.0f%%',startangle=90,textprops={'fontsize':8},
                    wedgeprops={'edgecolor':'white','linewidth':1.5})
            ax2.set_title("Répartition sujets",fontsize=9,fontweight='bold',color='#0f2347')
            ax2.set_facecolor('none')

            ax3=axes[2]
            at=["[Réf]"]+[f"#{i+1}" for i in range(len(recs))]
            as_=[int(ref['num_subscribers'])]+recs['num_subscribers'].astype(int).tolist()
            ac=['#ea580c']+['#93c5fd']*len(recs)
            ax3.barh(at[::-1],as_[::-1],color=ac[::-1],height=.6)
            ax3.set_title("Inscrits comparés",fontsize=9,fontweight='bold',color='#0f2347')
            ax3.set_xlabel("Inscrits",fontsize=8); ax3.tick_params(axis='y',labelsize=7.5)
            ax3.spines[['top','right']].set_visible(False); ax3.set_facecolor('none')

            plt.tight_layout(pad=1.5); st.pyplot(fig)
            st.markdown(f"""
            <div class="white-card" style="margin-top:.5rem;">
                <div style="font-size:.7rem;color:#2563eb;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:.6rem;">Résumé</div>
                <div style="font-size:.83rem;line-height:2.2;color:#374151;">
                    <b>Similarité moyenne :</b> <span style="font-family:'Syne',sans-serif;color:#2563eb;font-weight:800;">{recs['similarity'].mean()*100:.1f}%</span><br>
                    <b>Inscrits moyens (recs) :</b> <span style="color:#0891b2;font-weight:700;">{recs['num_subscribers'].mean():,.0f}</span><br>
                    <b>Sujets couverts :</b> <span style="color:#6b7280;">{recs['subject'].nunique()}</span><br>
                    <b>Cours gratuits :</b> <span style="color:#6b7280;">{(recs['price']==0).sum()} / {len(recs)}</span>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="sec-head">📈 Aperçu Catalogue <div class="sec-line"></div></div>', unsafe_allow_html=True)
        ca2,cb2=st.columns(2)
        with ca2:
            fig,ax=plt.subplots(figsize=(6,4))
            t10=df.nlargest(10,'num_subscribers')[['course_title','num_subscribers']]
            t10['s']=t10['course_title'].str[:28]+'…'
            ax.barh(t10['s'][::-1],t10['num_subscribers'][::-1],color='#2563eb',height=.65)
            ax.set_title("Top 10 — Plus d'inscrits",fontsize=10,fontweight='bold',color='#0f2347')
            ax.tick_params(axis='y',labelsize=7.5); ax.spines[['top','right']].set_visible(False)
            fig.patch.set_facecolor('none'); ax.set_facecolor('#f8faff')
            plt.tight_layout(); st.pyplot(fig)
        with cb2:
            fig,ax=plt.subplots(figsize=(6,4))
            sa=df.groupby('subject')['num_subscribers'].mean().sort_values()
            bars=ax.barh(sa.index,sa.values,color=['#0d9488','#7c3aed','#0891b2','#2563eb'],height=.55)
            ax.bar_label(bars,fmt='%.0f',fontsize=8,padding=3)
            ax.set_title("Inscrits moyens / Sujet",fontsize=10,fontweight='bold',color='#0f2347')
            ax.tick_params(axis='y',labelsize=8); ax.spines[['top','right']].set_visible(False)
            fig.patch.set_facecolor('none'); ax.set_facecolor('#f8faff')
            plt.tight_layout(); st.pyplot(fig)
        st.markdown("<div style='text-align:center;padding:2.5rem 1rem;color:#94a3c8;font-size:.95rem;'>☝️ Sélectionnez un cours puis cliquez sur <strong>Recommander</strong></div>", unsafe_allow_html=True)