# MovieLens Recommender (Python)

Ce dépôt contient un mini-système de recommandation de films basé sur le dataset **MovieLens** (GroupLens).  
Il est pensé pour être **reproductible**, **simple à utiliser**, et suffisamment “propre” pour un projet GitHub/CV.

Vous pouvez :
- télécharger MovieLens automatiquement,
- construire des objets de travail (matrice sparse + mappings),
- évaluer plusieurs modèles (baselines + collaborative filtering),
- générer des recommandations “top-K” pour un utilisateur donné.

---

## 1) Pré-requis

- Python (recommandé : **>= 3.10**)
- Git
- (Optionnel) VS Code

---

## 2) Installation (step-by-step)

### A. Cloner le repo
~~~bash
git clone https://github.com/<ton-user>/movielens-reco.git
cd movielens-reco
~~~

### B. Créer un environnement virtuel
~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

### C. Installer le projet
Si ton `pyproject.toml` contient les dépendances “dev” :
~~~bash
pip install -e ".[dev]"
~~~

Sinon (install minimal + outils dev séparés) :
~~~bash
pip install -e .
pip install numpy pandas scipy scikit-learn joblib tqdm pytest ruff
~~~

### D. Vérifier que tout marche
~~~bash
python -m reco health
pytest
ruff check .
~~~

---

## 3) Dataset MovieLens : où il est stocké ?

Le dataset est téléchargé dans :
- `data/ml-latest-small/` (par défaut)

Important :
- `data/` ne doit **pas** être versionné (il est dans `.gitignore`).
- Le README que tu vois dans `data/ml-latest-small/README.txt` vient du dataset, c’est normal.

---

## 4) Utilisation : commandes principales

Toutes les commandes passent par la CLI :
~~~bash
python -m reco <commande> [options]
~~~

### 4.1 Télécharger MovieLens
~~~bash
python -m reco download
~~~

### 4.2 Preprocess (stats + artefacts)
Cette étape charge `ratings.csv` et construit une matrice user-item (sparse), utile pour accélérer des traitements.
~~~bash
python -m reco preprocess
~~~

Résultats :
- affiche des stats (nb users, nb items, densité, etc.)
- écrit par défaut :
  - `artifacts/user_item_ratings_csr.npz`
  - `artifacts/mappings.json`

`artifacts/` est aussi à ignorer dans Git (souvent dans `.gitignore`).

---

## 5) Modèles disponibles

Le repo implémente 3 approches :

### A) Popularity baseline
- Recommande les films les plus “populaires” (ceux avec le plus de notes).
- Simple, mais utile comme baseline.

### B) Item-Item Cosine (collaborative filtering)
- On représente chaque film par un vecteur (les notes des utilisateurs).
- On calcule une similarité cosine entre films.
- Pour un utilisateur :
  - on prend ses derniers films notés (seeds),
  - on ajoute des scores aux voisins similaires,
  - pondération par la note (score += similarité × rating).

### C) SVD (matrix factorization)
- Factorise la matrice user-item en dimensions latentes.
- Recommande via produit scalaire (user_factors · item_factors).
- Souvent bon compromis : qualité + diversité (coverage).

---

## 6) Évaluation : comment ça marche ?

L’évaluation est faite via un **split temporel par utilisateur** :
- pour chaque utilisateur, on trie ses interactions par `timestamp`
- les **dernières** interactions vont en test (ex : 20%)
- le reste en train

Cela simule un cas réel : recommander des films “futurs” à partir de l’historique.

### Métriques (Top-K)
- **Precision@K** : proportion de recommandations pertinentes dans le top-K
- **Recall@K** : proportion des films pertinents retrouvés dans le top-K
- **NDCG@K** : qualité du classement (favorise les hits placés en haut)
- **Coverage** : diversité globale (part des items du train qui apparaissent au moins une fois recommandés)

---

## 7) Lancer les évaluations

### 7.1 Popularité
~~~bash
python -m reco evaluate --k 10 --test-ratio 0.2 --min-user-ratings 10
~~~

### 7.2 Item-item cosine
~~~bash
python -m reco evaluate-itemcos --k 10 --test-ratio 0.2 --min-user-ratings 10
~~~

Options utiles :
- `--n-seed` : nb d’items récents utilisés comme “seeds”
- `--n-neighbors` : nb de voisins par seed

### 7.3 SVD
~~~bash
python -m reco evaluate-svd --k 10 --test-ratio 0.2 --min-user-ratings 10 --n-components 50
~~~

---

## 8) Générer des recommandations (mode “démo”)

### 8.1 Item-item cosine
~~~bash
python -m reco recommend 1 --k 10
~~~

### 8.2 SVD
~~~bash
python -m reco recommend-svd 1 --k 10 --n-components 50
~~~

Tu verras parfois `(in test)` après un titre :
- cela signifie que le film recommandé fait partie des films “futurs” de l’utilisateur (dans le test split),
- c’est un bon signe : le modèle “retrouve” des films que l’utilisateur regardera plus tard.

---

## 9) Structure du projet (pour comprendre le code)

- `src/reco/cli.py`
  - point d’entrée CLI : parse les commandes et appelle les fonctions
- `src/reco/datasets.py`
  - téléchargement + extraction MovieLens
- `src/reco/preprocess.py`
  - charge `ratings.csv`, construit la matrice sparse et calcule les stats
- `src/reco/eval.py`
  - split temporel + métriques (precision/recall/ndcg)
- `src/reco/itemcos.py`
  - modèle item-item cosine (avec voisins pré-calculés pour aller vite)
- `src/reco/eval_itemcos.py`
  - évaluation du modèle itemcos
- `src/reco/svd.py`
  - modèle SVD (TruncatedSVD)
- `src/reco/eval_svd.py`
  - évaluation du modèle SVD
- `src/reco/recommend.py`
  - transforme des `movieId` en titres (`movies.csv`) et affiche un top-K

Tests :
- `tests/`
  - tests unitaires (split temporel, métriques, etc.)

---

## 10) Checklist “repo propre”
Avant de pousser :
~~~bash
pytest
ruff check .
~~~

---