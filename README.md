# 📈 News-Based Stock Market Prediction — DJIA

> Prédire la direction quotidienne du Dow Jones Industrial Average (hausse/baisse) à partir de titres de news Reddit et de features financières.

---

## 1. Informations du projet

| Champ | Détail |
|---|---|
| **Titre** | News-Based Stock Market Prediction — DJIA |
| **Cours** | AI In Finance |
| **Encadrants** | Nicolas De Roux & Mohamed El Fakir |
| **Collaborateurs** | Akram Boudebouze, Alexander Besson, Ferdinand Carsoux, Raphaël Essengue |

---

## 2. Description du projet

Les marchés financiers sont influencés non seulement par des données chiffrées, mais aussi par le flux d'informations qui circule chaque jour dans les médias. Ce projet explore dans quelle mesure les **titres de news** publiés sur Reddit (subreddit WorldNews) permettent de prédire la **direction journalière du DJIA** (hausse ou baisse), en les combinant à des **indicateurs financiers** classiques et à une analyse de **sentiment**.

La question est à la fois économiquement pertinente — pour tout acteur de marché cherchant à intégrer l'information non-structurée dans ses modèles — et techniquement stimulante, car elle croise NLP, feature engineering financier et séries temporelles.

---

## 3. Objectif

Construire un **classifieur binaire** capable de prédire, pour chaque jour de bourse, si le DJIA clôture en hausse (1) ou en baisse (0), à partir des 25 titres Reddit du jour et de features financières calculées sur les cours historiques.

Une solution performante est un modèle dont l'**AUC-ROC** dépasse significativement 0.50 (le hasard), de façon **stable dans le temps** — ce que nous évaluons via une validation walk-forward.

---

## 4. Définition de la tâche

| Élément | Détail |
|---|---|
| **Type de tâche** | Classification binaire supervisée |
| **Input** | 25 titres Reddit concaténés + features financières (rendements, volatilité, RSI) + scores de sentiment VADER |
| **Output** | `Label` : 1 (hausse DJIA) / 0 (baisse DJIA) |
| **Métriques** | Accuracy, F1-score, AUC-ROC |

---

## 5. Description du dataset

### Vue d'ensemble

| Propriété | Valeur |
|---|---|
| **Source** | [Combined News DJIA — Kaggle](https://www.kaggle.com/datasets/aaron7sun/stocknews) |
| **Période** | Août 2008 → Juillet 2016 (8 ans) |
| **Nombre de lignes** | ~1 989 jours de bourse |
| **Nombre de colonnes** | 27 (Date, Label, Top1 à Top25) |
| **Variable cible** | `Label` (binaire) |

### Description des features

| Feature | Description | Type |
|---|---|---|
| `Date` | Date du jour de bourse | Temporel |
| `Label` | Hausse (1) ou baisse (0) du DJIA | Binaire |
| `Top1` à `Top25` | 25 titres Reddit/WorldNews du jour | Texte |
| `ret_1d` | Rendement logarithmique J-1 | Numérique |
| `ret_5d` | Rendement cumulé sur 5 jours | Numérique |
| `vol_5d` / `vol_20d` | Volatilité glissante (5j / 20j) | Numérique |
| `rsi_14` | RSI sur 14 jours | Numérique |
| `day_of_week` | Jour de la semaine (0=lundi…4=vendredi) | Catégoriel |
| `sent_compound` | Score de sentiment VADER agrégé | Numérique |

### Variable cible

`Label` est une variable binaire : **1** si le DJIA clôture en hausse par rapport à la veille, **0** sinon. La distribution est quasi-équilibrée : 53.5% de jours en hausse, 46.5% en baisse.

### Distribution des données

- **Équilibre des classes** : quasi-parfait (53.5% / 46.5%) — pas de rééchantillonnage nécessaire.
- **Longueur des textes** : ~500 mots par jour en moyenne (25 titres × ~20 mots).
- **Vocabulaire dominant** : termes géopolitiques (police, china, government, iran, sanctions) — cohérent avec le corpus Reddit/WorldNews.
- **Crise 2008–2009** : clairement visible dans les cours DJIA (chute de ~12k à ~6.5k).

### Qualité des données

- **Bug b'...'** : les titres issus d'un scraping Python 2 contiennent des préfixes parasites (`b"..."`) — corrigés via nettoyage regex.
- **Valeurs manquantes** : aucune après nettoyage.
- **Doublons** : absents.

---

## 6. Prétraitement des données

| Étape | Description |
|---|---|
| **Nettoyage texte** | Suppression des préfixes `b'...'` hérités du scraping Python 2 via regex |
| **Concaténation** | Les 25 titres par jour sont fusionnés en un seul texte `texte_complet` |
| **Split chronologique** | Train (2008–2013), Validation (2014), Test (2015–2016) — aucun data leakage temporel |
| **TF-IDF** | Vectorisation (max 5 000 features, bigrammes, `sublinear_tf`) — fitté sur train uniquement |
| **Features financières** | Calcul via `yfinance` : rendements, volatilité, RSI — normalisés avec `StandardScaler` fitté sur train |
| **Sentiment VADER** | Score compound par jour calculé sur le texte concaténé |
| **Word2Vec** | Embeddings entraînés sur le corpus train (dim=100, window=5) — moyenne par jour |

---

## 7. Approche de modélisation

### Modèles utilisés

Le projet est structuré en **3 niveaux de complexité croissante** :

| Niveau | Représentation textuelle | Modèles |
|---|---|---|
| **N1 — Classique** | TF-IDF + features financières + VADER | Logistic Regression, Random Forest, Gradient Boosting |
| **N2 — Embeddings** | Word2Vec (mean pooling) + features financières + VADER | Gradient Boosting |
| **N3 — Transformer** | FinBERT (fine-tuné) | FinBERT (ProsusAI) |

### Stratégie de modélisation

- **Baseline** : Régression Logistique (TF-IDF seul).
- **Hyperparamètres** : fixés manuellement à des valeurs standards reconnues dans la littérature (ex. GB : `n_estimators=200`, `lr=0.05`, `max_depth=4`).
- **Pas de grid search** : le signal intrinsèque du dataset est trop faible pour que le tuning apporte un gain substantiel.
- **Validation walk-forward** : évaluation glissante par fenêtres annuelles pour tester la stabilité temporelle.
- **Pas de data leakage** : tous les transformateurs (TF-IDF, Scaler, Word2Vec) sont fittés sur le train uniquement.

### Métriques d'évaluation

| Métrique | Justification |
|---|---|
| **AUC-ROC** | Métrique principale — mesure la capacité discriminante indépendamment du seuil et du biais de classe |
| **Accuracy** | Indicateur global, mais trompeur en cas de biais de prédiction |
| **F1-score** | Sensible au biais vers la classe majoritaire — à interpréter avec l'AUC |

---

## 8. Résultats

| Modèle | Accuracy | F1 | AUC-ROC | Commentaire |
|---|---|---|---|---|
| Logistique (N1) | 0.487 | 0.592 | 0.481 | Sous le hasard en accuracy |
| Random Forest (N1) | 0.508 | 0.652 | 0.481 | Meilleure accuracy N1, AUC identique |
| **Gradient Boosting (N1)** | **0.511** | 0.588 | **0.537** | Meilleur AUC du N1 |
| **Word2Vec + GB (N2)** | **0.534** | 0.627 | **0.550** | ✅ **Meilleur modèle global** |
| FinBERT (N3) | 0.503 | 0.665 | 0.482 | F1 élevé mais AUC quasi-aléatoire (overfitting) |

**Conclusion** : Word2Vec + Gradient Boosting est le meilleur modèle (AUC 0.550, Accuracy 0.534). FinBERT sous-performe faute d'un corpus suffisamment large pour fine-tuner 109M paramètres. Le signal texte est principalement un **signal de risque géopolitique** (baisse) plutôt qu'un signal haussier structuré.

---

## 9. Structure du projet

```
.
├── DJIA_NewsPrediction_VF.ipynb   # Notebook principal (exploration → modélisation → interprétation)
├── Combined_News_DJIA.csv         # Dataset source (à télécharger sur Kaggle)
├── requirements.txt               # Dépendances Python
├── etape1_visualisation.png       # Graphiques EDA
├── etape3_comparaison.png         # Comparaison des modèles
└── etape4_interpretation.png      # Feature importance & coefficients
```

---

## 10. Installation

```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/DJIA-NewsPrediction.git
cd DJIA-NewsPrediction

# Installer les dépendances
pip install -r requirements.txt
```

**Ou directement dans le notebook :**

```bash
pip install yfinance vaderSentiment gensim sentence-transformers transformers torch scikit-learn pandas matplotlib seaborn
```

> ⚠️ Le dataset `Combined_News_DJIA.csv` est à télécharger manuellement depuis [Kaggle](https://www.kaggle.com/datasets/aaron7sun/stocknews) et à placer à la racine du dépôt.
