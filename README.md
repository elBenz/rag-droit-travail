# RAG droit du travail

Prototype de **Retrieval-Augmented Generation** qui répond à des questions sur le droit du travail
français à partir des 612 fiches officielles Service-Public du thème « Travail - Formation ».
Chaque réponse cite les fiches utilisées. Si l'information n'est pas dans le corpus, le système le dit
au lieu d'inventer.

```
POST /ask {"question": "Quelle est la durée maximale de la période d'essai d'un cadre en CDI ?"}
→ {"answer": "… 4 mois [1] …", "sources": [{"id": "F1643", "title": "Période d'essai pour un salarié", "url": "https://www.service-public.gouv.fr/…/F1643"}]}
```

## Architecture

```
                 INGESTION (une fois)                                 QUESTION (à chaque requête)

 XML Service-Public ──► parsing ──► découpage ──► embeddings ──► pgvector ◄── recherche des 30 plus proches ◄── embedding de la question
  (612 fiches)        sections    10 101 chunks   Voyage 1024 d.  (HNSW)          │
                                                                                  ▼
                                                                  reranking Voyage rerank-2.5 → 5 meilleurs
                                                                                  │
                                                                                  ▼
                                                              extraits numérotés [1]…[5] ──► Claude ──► réponse + sources citées
```

| Module | Rôle |
|---|---|
| `rag/parsing.py` | XML → fiche (id, titre, URL) et sections avec leur chemin de titres (situation > chapitre > cas). Les tableaux restent sur une ligne par rangée. |
| `rag/chunking.py` | Une section = un chunk, redécoupé au-delà de 1 500 caractères (chevauchement de 200). Chaque chunk commence par son contexte : `Période d'essai > CDI > Quelle est la durée… > Cadre`. |
| `rag/store.py` | Index pgvector (table `chunks`, index HNSW cosinus) et recherche des `top_k` plus proches. |
| `rag/rerank.py` | Recherche large (30 candidats) puis reranking par un cross-encoder, qui garde les `top_k` meilleurs. |
| `rag/answer.py` | Prompt avec extraits numérotés, consignes anti-hallucination, extraction des sources réellement citées. |
| `rag/api.py` | API FastAPI `POST /ask`. |
| `rag/evaluate.py` | Évaluation sur `eval/questions.jsonl`. |

Stack : Python 3.13, FastAPI, LangChain (`langchain-anthropic`, `langchain-voyageai`, découpeur de texte), PostgreSQL 17 + pgvector, Claude (`claude-opus-5-5`) pour la génération, Voyage AI pour les embeddings (`voyage-3.5`) et le reranking (`rerank-2.5`).

## Les notions, et les choix faits ici

- **Embedding** : un vecteur qui représente le sens d'un texte. Deux textes proches par le sens ont des
  vecteurs proches, même sans mot commun (« prime de précarité » ↔ « indemnité de fin de contrat »).
  Voyage distingue l'embedding d'un document et celui d'une question (`input_type`) : c'est un modèle asymétrique.
  Anthropic ne propose pas d'embeddings ; Voyage est le partenaire qu'il recommande.
- **Chunking (découpage)** : on n'indexe pas une fiche entière (trop large, le sens se dilue) ni une phrase
  (trop peu de contexte). Ici, le découpage suit la **structure de la fiche** (un chunk par cas) plutôt qu'une
  taille fixe, et chaque chunk porte son chemin de titres. Sans ça, « 4 mois » ne dirait pas qu'il s'agit
  d'un cadre en CDI. Les fiches « conditionnées » (CDI / CDD / intérim) étaient au départ ignorées par le
  parseur : c'est le contrôle sur le vrai corpus qui l'a révélé (4 453 chunks au lieu de 10 101).
- **Base vectorielle** : pgvector garde vecteurs et métadonnées dans PostgreSQL, sans service en plus.
  L'index **HNSW** fait une recherche approximative des plus proches voisins, rapide même sur des millions de vecteurs.
- **top-k** : le nombre d'extraits donnés au LLM (5 par défaut). Trop petit, la bonne information manque ;
  trop grand, on paie plus de tokens et on noie le modèle sous du bruit.
- **Hallucinations** : le LLM peut inventer une réponse plausible. Garde-fous : consigne de ne répondre
  qu'à partir des extraits, citation obligatoire `[n]`, phrase de refus imposée, et sources calculées à partir
  des citations réelles (pas déclarées par le modèle). Le jeu d'évaluation contient des questions hors corpus
  pour vérifier le refus.
- **Reranking** : la recherche vectorielle compare deux vecteurs calculés séparément, ce qui est rapide
  mais grossier. Pour « Je suis cadre en CDI, combien de temps peut durer ma période d'essai ? », le bon
  extrait arrivait 6e, à 0,004 de similarité du 1er, derrière des fiches CDD et fonction publique : la
  réponse disait alors ne pas savoir pour le privé. Un reranker (cross-encoder, `rerank-2.5`) lit la
  question et chaque candidat **ensemble** et note leur pertinence. Il est trop lent pour tout le corpus,
  d'où le schéma en deux temps : 30 candidats par la recherche, 5 gardés après reranking.
- **Recherche hybride** (non implémentée) : combiner vecteurs et mots-clés (`tsvector` de PostgreSQL,
  BM25) aide sur les termes exacts : numéros d'article, sigles.
- **Évaluation** : on mesure séparément la **recherche** (la bonne fiche est-elle dans le top-k ? hit@k,
  MRR) et la **génération** (la réponse contient-elle l'élément attendu ? cite-t-elle la bonne fiche ?).
  Une mauvaise réponse vient souvent d'une mauvaise recherche : séparer les deux dit quoi corriger.

## Lancer le projet

Prérequis : Docker, [uv](https://docs.astral.sh/uv/), une clé API Anthropic et une clé Voyage AI.

```bash
cp .env.example .env          # puis renseigner ANTHROPIC_API_KEY et VOYAGE_API_KEY
docker compose up -d          # PostgreSQL + pgvector sur le port 5433
uv sync

# Corpus : archive open data Service-Public (≈ 23 Mo, licence ouverte)
mkdir -p data/raw && curl -o data/raw/vosdroits.zip https://lecomarquage.service-public.fr/vdd/3.4/part/zip/vosdroits-latest.zip
unzip -q data/raw/vosdroits.zip -d data/raw/vosdroits

uv run python -m rag.ingest data/raw/vosdroits    # parsing, chunks, embeddings, index
uv run uvicorn rag.api:app --reload               # http://localhost:8000/docs
uv run python -m rag.evaluate 5 --rerank          # évaluation (top_k = 5, avec reranking ; sans --rerank pour comparer)
uv run pytest                                     # tests (sans appel d'API)
```

Les tests n'appellent aucune API : ils utilisent des embeddings factices déterministes et un faux LLM,
et une base PostgreSQL de test (`rag_test`) recréée à chaque test.

## Résultats de l'évaluation

13 questions dont la réponse est vérifiée dans les fiches, plus 2 questions hors corpus, top_k = 5
(`eval/results-baseline.txt`, `eval/results-rerank.txt`) :

| | Sans reranking | Avec reranking |
|---|---|---|
| hit@5 (bonne fiche dans les 5 extraits) | 12/13 | **13/13** |
| MRR (rang moyen inverse) | 0,88 | **0,96** |
| Réponse contenant l'élément attendu | 13/13 | 13/13 |
| Fiche attendue citée | 12/13 | **13/13** |
| Refus hors corpus | 2/2 | 2/2 |

Ce que l'évaluation a appris :
- Sans reranking, la question sur le congé de maternité (qui ne précise pas « secteur privé ») ne
  remontait que la fiche de la fonction publique. La réponse (16 semaines) restait juste, car les
  deux secteurs ont la même durée, mais la source n'était pas la bonne. Le reranking remonte la fiche
  du privé en 2e position.
- Le jeu de test lui-même avait une erreur : pour la prime de précarité, la fiche F803 répond aussi
  exactement à la question. Elle a été ajoutée aux fiches attendues, et les deux versions ont été
  réévaluées sur le jeu corrigé.
- Coût indicatif : l'indexation (≈ 2 M tokens) tient dans l'offre gratuite de Voyage ; une question
  coûte de l'ordre d'un à deux centimes de Claude.

## Limites

- Petit jeu d'évaluation (13 questions + 2 hors corpus), vérification par mots-clés : utile pour comparer
  deux versions, pas une mesure absolue. Il faudrait des questions plus difficiles (multi-fiches,
  formulations familières) et une notation par un LLM juge pour aller plus loin.
- Les fiches du secteur public (fonction publique) sont dans le corpus : une question ambiguë peut remonter
  la mauvaise fiche.
- Pas de mémoire de conversation, pas de mise à jour incrémentale (ré-ingestion complète).
- Ce n'est pas un conseil juridique : les réponses renvoient vers les fiches officielles.
