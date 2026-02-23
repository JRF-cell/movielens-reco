# MovieLens Recommender (Python)

Système de recommandation de films basé sur MovieLens (GroupLens), avec baselines + collaborative filtering, et une évaluation reproductible.

## Stack
- Python (venv)
- numpy / pandas / scipy
- scikit-learn
- CLI : `python -m reco ...`

## Installation
~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install numpy pandas scipy scikit-learn joblib tqdm pytest ruff
~~~

## Usage

### 1) Télécharger le dataset
~~~bash
python -m reco download
~~~

### 2) Preprocess (matrice sparse + mappings)
~~~bash
python -m reco preprocess
~~~

### 3) Évaluer

Baseline popularité :
~~~bash
python -m reco evaluate --k 10 --test-ratio 0.2 --min-user-ratings 10
~~~

Item-item cosine :
~~~bash
python -m reco evaluate-itemcos --k 10 --test-ratio 0.2 --min-user-ratings 10
~~~

## Results (ml-latest-small)
Temporal split per user: 80/20, K=10, min_user_ratings=10

| Model | Precision@10 | Recall@10 | NDCG@10 |
|------|--------------:|----------:|--------:|
| Popularity | 0.0718 | 0.0388 | 0.0865 |
| Item-Item Cosine | 0.0921 | 0.0623 | 0.1034 |
