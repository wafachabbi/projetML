# Rapport Final — Smartek AI
## Plateforme Intelligente de Gestion des Formations Udemy
### Approche CRISP-DM

---

## 1. Introduction

Ce rapport présente les travaux réalisés dans le cadre du projet **Smartek AI**, une plateforme d'analyse intelligente des cours de la plateforme Udemy. Le projet s'inscrit dans une démarche **CRISP-DM** (Cross-Industry Standard Process for Data Mining), méthodologie de référence pour les projets de Data Science.

L'objectif principal est de fournir à une organisation de formation en ligne des outils d'aide à la décision basés sur l'intelligence artificielle, afin d'optimiser la création de contenu, l'organisation du catalogue et l'expérience utilisateur.

Le projet couvre trois axes fonctionnels :
- **Prédiction de popularité** des cours (classification)
- **Segmentation automatique** des cours (clustering)
- **Recommandation personnalisée** de cours

---

## 2. Compréhension Métier (Business Understanding)

### 2.1 Contexte

Le marché des formations en ligne est en forte croissance. Les plateformes comme Udemy proposent des milliers de cours dans des domaines variés. Face à cette abondance, les créateurs de contenu et les équipes marketing ont besoin d'outils pour :
- Identifier les formations à fort potentiel avant leur lancement
- Mieux organiser le catalogue pour faciliter la navigation
- Proposer des recommandations pertinentes aux apprenants

### 2.2 Objectifs Métier (Business Objectives)

| # | Objectif |
|---|----------|
| **BO 1** | Identifier les formations à fort potentiel de succès afin d'optimiser la stratégie de création de contenu et l'allocation marketing. |
| **BO 2** | Regrouper les formations similaires afin d'améliorer l'organisation du catalogue et faciliter la recherche des utilisateurs. |
| **BO 3** | Proposer aux apprenants des formations pertinentes selon le sujet, le niveau et la popularité afin d'améliorer l'expérience utilisateur et d'augmenter les inscriptions. |

### 2.3 Objectifs Data Science (DSOs)

| # | Objectif Data Science | Lien BO |
|---|----------------------|---------|
| **DSO 1** | Construire un modèle de **classification binaire** capable de prédire si un cours sera populaire ou non à partir de ses caractéristiques (prix, durée, nombre de leçons, avis, sujet, niveau). | BO 1 |
| **DSO 2** | Appliquer un algorithme de **clustering (KMeans)** pour regrouper automatiquement les cours en segments homogènes, optimisé par le score de silhouette. | BO 2 |
| **DSO 3** | Développer un **moteur de recommandation** basé sur le contenu (TF-IDF + KNN cosinus) pour suggérer des cours similaires à un cours donné. | BO 3 |

---

## 3. Compréhension des Données (Data Understanding)

### 3.1 Source des données

Le dataset utilisé est **udemy_courses.csv**, un jeu de données public contenant des informations sur les cours proposés sur la plateforme Udemy.

### 3.2 Description du dataset

| Caractéristique | Valeur |
|----------------|--------|
| Nombre de lignes | **3 678 cours** |
| Nombre de colonnes | **12 variables** |
| Valeurs manquantes | **0** (dataset complet) |

### 3.3 Variables disponibles

| Variable | Type | Description |
|----------|------|-------------|
| `course_id` | int | Identifiant unique du cours |
| `course_title` | str | Titre du cours |
| `url` | str | Lien vers le cours |
| `is_paid` | bool | Cours payant ou gratuit |
| `price` | int | Prix en dollars (0 si gratuit) |
| `num_subscribers` | int | Nombre d'inscrits |
| `num_reviews` | int | Nombre d'avis |
| `num_lectures` | int | Nombre de leçons |
| `level` | str | Niveau (All Levels, Beginner, Intermediate, Expert) |
| `content_duration` | float | Durée totale en heures |
| `published_timestamp` | str | Date de publication |
| `subject` | str | Catégorie (Web Development, Business Finance, Musical Instruments, Graphic Design) |

### 3.4 Statistiques descriptives

**Variables numériques :**

| Variable | Moyenne | Médiane | Min | Max |
|----------|---------|---------|-----|-----|
| Prix ($) | 66.05 | 45 | 0 | 200 |
| Inscrits | 3 197 | 911 | 0 | 268 923 |
| Avis | 156 | 18 | 0 | 27 445 |
| Leçons | 40 | 25 | 0 | 779 |
| Durée (h) | 4.09 | 2.0 | 0 | 78.5 |

**Variables catégorielles :**
- **4 catégories** : Web Development (1 200 cours), Business Finance, Musical Instruments, Graphic Design
- **4 niveaux** : All Levels (1 929 cours), Beginner Level, Intermediate Level, Expert Level

### 3.5 Observations clés

- La distribution des inscrits est **très asymétrique** : la médiane (911) est bien inférieure à la moyenne (3 197), indiquant quelques cours très populaires qui tirent la moyenne vers le haut.
- **8.4%** des cours sont gratuits.
- Web Development est la catégorie la plus représentée avec 1 200 cours.
- Aucune valeur manquante n'a été détectée, ce qui simplifie la préparation des données.

---

## 4. Préparation des Données (Data Preparation)

### 4.1 Traitement des valeurs aberrantes

Pour chaque variable numérique (`price`, `num_subscribers`, `num_reviews`, `num_lectures`, `content_duration`), un **écrêtage IQR** a été appliqué :
- Calcul du 1er quartile (Q1) et du 3ème quartile (Q3)
- Écrêtage des valeurs en dehors de l'intervalle `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`

Cette technique permet de réduire l'impact des valeurs extrêmes sans supprimer de données.

### 4.2 Ingénierie des features

Deux nouvelles variables ont été créées pour enrichir l'analyse :

| Feature | Formule | Interprétation |
|---------|---------|----------------|
| `reviews_per_sub` | `num_reviews / num_subscribers` | Taux d'engagement des apprenants |
| `lectures_per_hour` | `num_lectures / content_duration` | Densité du contenu |

### 4.3 Encodage des variables catégorielles

Pour le modèle de classification (DSO 1) :
- **One-Hot Encoding** appliqué aux colonnes `subject` et `level`
- La variable `is_paid` convertie en entier (0/1)

### 4.4 Normalisation (pour le clustering)

Pour le clustering (DSO 2) :
- **StandardScaler** appliqué sur les 6 features numériques pour ramener toutes les variables à la même échelle

### 4.5 Préparation pour la recommandation

Pour le moteur de recommandation (DSO 3) :
- Création d'une colonne `combined` = `course_title` + `subject` + `level`
- Vectorisation TF-IDF avec 5 000 features et n-grammes (1,2)

---

## 5. Modélisation (Modeling)

### 5.1 DSO 1 — Classification Binaire (XGBoost)

**Objectif :** Prédire si un cours sera populaire (nombre d'inscrits ≥ médiane).

**Définition de la cible :**
- Seuil de popularité = **médiane du nombre d'inscrits** (~911 inscrits)
- Classe 1 : cours populaire (≥ seuil)
- Classe 0 : cours peu populaire (< seuil)

**Algorithme :** XGBoost Classifier

**Hyperparamètres :**
```
n_estimators    = 200
max_depth       = 5
learning_rate   = 0.1
subsample       = 0.8
colsample_bytree = 0.8
```

**Division des données :** 80% entraînement / 20% test

**Résultats :**

| Métrique | Score |
|----------|-------|
| Accuracy | ~85% |
| F1-Score | ~84% |
| AUC-ROC | ~0.92 |

**Facteurs clés de succès identifiés :**
1. Nombre d'avis (`num_reviews`)
2. Taux d'engagement (`reviews_per_sub`)
3. Nombre de leçons (`num_lectures`)
4. Durée du cours (`content_duration`)
5. Prix (`price`)

---

### 5.2 DSO 2 — Clustering (KMeans)

**Objectif :** Segmenter les cours en groupes homogènes.

**Features utilisées :**
- `price`, `num_reviews`, `num_lectures`, `content_duration`, `reviews_per_sub`, `lectures_per_hour`

**Algorithme :** KMeans avec optimisation du nombre de clusters K

**Sélection du K optimal :**
- Test de K = 2 à 8
- Sélection basée sur le **score de silhouette** (mesure la qualité de la séparation entre clusters)
- K optimal retenu : **4 clusters**
- Score de silhouette : ~0.35

**Visualisation :** Réduction dimensionnelle par **PCA (2 composantes)** pour visualiser les clusters.

**Profil des groupes identifiés :**

| Groupe | Caractéristiques principales |
|--------|------------------------------|
| Groupe 1 | Cours courts, peu chers, peu d'avis → Cours débutants/découverte |
| Groupe 2 | Cours longs, prix élevé, beaucoup de leçons → Formations complètes |
| Groupe 3 | Cours populaires, beaucoup d'avis → Best-sellers |
| Groupe 4 | Cours gratuits ou très peu chers → Contenu d'appel |

---

### 5.3 DSO 3 — Recommandation (TF-IDF + KNN)

**Objectif :** Proposer des cours similaires à un cours donné.

**Approche :** Système de recommandation basé sur le contenu (Content-Based Filtering)

**Pipeline :**
1. Création d'un texte combiné : `titre + sujet + niveau`
2. Vectorisation **TF-IDF** (5 000 features, n-grammes 1-2, stop words anglais supprimés)
3. Modèle **KNN** avec métrique cosinus (algorithme brute, 11 voisins)
4. Pour une requête : calcul de la similarité cosinus avec tous les cours, retour des N plus proches

**Avantages de cette approche :**
- Ne nécessite pas d'historique utilisateur (cold start résolu)
- Résultats interprétables
- Rapide à l'inférence

---

## 6. Déploiement (Deployment)

### 6.1 Application Web — Smartek AI Dashboard

L'ensemble des modèles a été déployé sous forme d'une **application web interactive** développée avec **Streamlit** et hébergée sur **Streamlit Community Cloud**.

**URL de déploiement :** https://projetml-4dbfsvcfrpgttmyueenur7.streamlit.app

**Technologies utilisées :**

| Composant | Technologie |
|-----------|-------------|
| Interface web | Streamlit |
| Modèles ML | Scikit-learn, XGBoost |
| Sérialisation | Joblib (.pkl) |
| Hébergement | Streamlit Community Cloud |
| Versioning | GitHub (wafachabbi/projetML) |

### 6.2 Architecture de l'application

```
projetML/
├── App.py                    # Application principale Streamlit
├── requirements.txt          # Dépendances Python
├── udemy_courses.csv         # Dataset
├── models/
│   ├── dso1_classifier.pkl   # Modèle XGBoost (classification)
│   ├── dso1_feature_cols.pkl # Colonnes features
│   ├── dso1_threshold.pkl    # Seuil de popularité
│   ├── dso2_kmeans.pkl       # Modèle KMeans (clustering)
│   ├── dso2_scaler.pkl       # StandardScaler
│   ├── dso3_tfidf.pkl        # Vectoriseur TF-IDF
│   ├── dso3_knn.pkl          # Modèle KNN
│   └── dso3_tfidf_matrix.pkl # Matrice TF-IDF précalculée
└── notebook/
    └── ML_CRISP_DM_4SAE1.ipynb  # Notebook d'analyse
```

### 6.3 Fonctionnalités du Dashboard

L'application est organisée en **4 pages** accessibles via la barre de navigation :

#### Page 1 — Vue Générale
- KPIs globaux : nombre de cours, inscrits moyens, avis moyens, % gratuits
- Présentation des 3 fonctionnalités IA
- Visualisations : répartition par sujet, par niveau, inscrits moyens par sujet
- Top 10 des cours les plus populaires

#### Page 2 — Prédiction de Popularité
- Formulaire interactif : saisie des caractéristiques d'un cours (sujet, niveau, prix, durée, leçons, avis estimés)
- Résultat de prédiction avec indice de confiance
- Visualisation des facteurs clés de succès
- Distribution des inscriptions avec seuil de popularité

#### Page 3 — Segmentation des Cours
- Carte interactive des groupes de cours (visualisation PCA 2D)
- Taille de chaque groupe
- Profil moyen par groupe (prix, avis, leçons, durée, inscrits)
- Répartition des catégories par groupe

#### Page 4 — Suggestions de Cours
- Filtres : catégorie, niveau, type (payant/gratuit)
- Sélection d'un cours de référence
- Affichage des N cours les plus similaires avec taux de ressemblance
- Graphique des scores de similarité

### 6.4 Performance et optimisation

- Les modèles sont **chargés une seule fois** au démarrage grâce au cache Streamlit (`@st.cache_resource`)
- La matrice TF-IDF est **précalculée** et sauvegardée pour éviter le recalcul à chaque requête
- Le dataset est mis en cache (`@st.cache_data`) pour éviter les lectures répétées

---

## 7. Conclusion et Perspectives

### 7.1 Bilan du projet

Le projet Smartek AI a permis de développer une plateforme complète d'analyse intelligente des cours Udemy, répondant aux trois objectifs métier définis :

| Objectif | Résultat |
|----------|----------|
| **BO 1** — Identifier les formations à fort potentiel | ✅ Modèle XGBoost avec ~85% de précision |
| **BO 2** — Regrouper les formations similaires | ✅ 4 groupes cohérents identifiés par KMeans |
| **BO 3** — Proposer des formations pertinentes | ✅ Moteur de recommandation TF-IDF + KNN opérationnel |

### 7.2 Points forts

- **Dataset propre** : aucune valeur manquante, facilitant la préparation
- **Modèles performants** : XGBoost atteint ~85% de précision avec un AUC-ROC de ~0.92
- **Application déployée** : accessible en ligne, interface intuitive orientée client
- **Architecture modulaire** : modèles sauvegardés séparément, facilement remplaçables

### 7.3 Limites

- Le dataset est limité à **4 catégories** de cours Udemy, ce qui restreint la généralisation
- Le modèle de recommandation est **basé uniquement sur le contenu textuel** (titre, sujet, niveau) et ne prend pas en compte le comportement des utilisateurs
- Les scores de performance (~85%) pourraient être améliorés avec plus de données ou un tuning plus poussé des hyperparamètres

### 7.4 Perspectives d'amélioration

1. **Enrichissement des données** : intégrer des données comportementales (clics, temps passé, taux de complétion) pour améliorer les recommandations
2. **Filtrage collaboratif** : combiner le filtrage basé sur le contenu avec un filtrage collaboratif (comportement des utilisateurs similaires)
3. **Mise à jour automatique** : pipeline de réentraînement automatique des modèles lorsque de nouvelles données sont disponibles
4. **Personnalisation** : adapter les recommandations au profil de chaque apprenant (historique, préférences, niveau)
5. **Extension du catalogue** : étendre l'analyse à d'autres plateformes (Coursera, edX) pour une vision plus large du marché
6. **A/B Testing** : mesurer l'impact réel des recommandations sur les taux d'inscription

---

*Rapport rédigé dans le cadre du projet Smartek AI — Approche CRISP-DM*
