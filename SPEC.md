# Spécification — RAG droit du travail

## But

Répondre en français à des questions sur le droit du travail (secteur privé, côté salarié et employeur)
à partir des fiches officielles Service-Public, **en citant les fiches utilisées**, et refuser de
répondre quand le corpus ne contient pas l'information.

## Corpus

- Source : archive open data `vosdroits-latest.zip` (lecomarquage.service-public.fr, schéma 3.4, licence ouverte).
- Périmètre : les fiches `F*.xml` du thème `N19806` « Travail - Formation » (671 fiches au 24/11/2026).
- Une fiche = un identifiant (`F2258`), un titre, une URL publique (`spUrl`), une introduction, des chapitres.
  Certains chapitres se déclinent en « cas » (ex. « Cas général », « Apprenti »).

## Pipeline

1. **Parsing** (`rag.parsing`) : XML → `Fiche(id, title, url, sections)`, où chaque section a un chemin
   de titres (`["Quel salarié a droit aux congés payés ?", "Apprenti"]`) et un texte brut sans balises.
2. **Découpage** (`rag.chunking`) : une section = un chunk ; une section trop longue (> `max_chars`) est
   redécoupée avec chevauchement. Chaque chunk commence par son contexte
   (`Titre de la fiche > Chapitre > Cas`) pour rester compréhensible seul.
3. **Indexation** (`rag.store`) : embeddings Voyage (`voyage-3.5`, 1024 dimensions) stockés dans
   PostgreSQL + pgvector, avec l'identifiant, le titre et l'URL de la fiche. Ré-indexer remplace l'existant.
4. **Recherche** : similarité cosinus sur 30 candidats, puis reranking (`rerank-2.5`), qui garde `top_k` chunks (défaut 5).
5. **Génération** (`rag.answer`) : Claude reçoit la question et les chunks numérotés `[1]…[k]`.
   Il doit répondre uniquement à partir de ces extraits, citer les numéros utilisés, et répondre
   « Je ne trouve pas cette information dans les fiches. » sinon. La réponse renvoie les sources citées
   (id, titre, URL), dédoublonnées.
6. **API** (`rag.api`) : `POST /ask {"question": str, "top_k"?: int}` →
   `{"answer": str, "sources": [{"id", "title", "url"}]}`. Question vide → 422.

## Évaluation

`eval/questions.jsonl` : questions avec la ou les fiches attendues et des éléments de réponse attendus.
- **Recherche** : taux de questions dont une fiche attendue figure dans le top-k (hit@k), et MRR.
- **Réponse** : présence des éléments attendus (vérification par mots-clés) et citation d'une fiche attendue.
- Questions hors corpus : la réponse doit être le refus.

## Hors périmètre (pour l'instant)

Recherche hybride (mots-clés + vecteurs), historique de conversation, interface web.
