<!--
Paste everything under the title into the Google Doc.
Redraw Figure 1, Figure 2, and Figure 3 in Excalidraw.
Do not paste the "Cuts for LinkedIn, Hashnode, and X" section into Medium.
This draft follows the narrative shape of the sample pieces: a concrete failure, a single technical hero, an architecture you can redraw, then a pipeline explained stage by stage with the code that implements it.
Target length: about 2200–3500 words, plus code blocks.
-->

# Beyond Keyword Search: Hybrid Retrieval for EU Law with Qdrant

## Introduction and the LegalTech challenge

Consider this. A regulation changes on a Thursday. By Friday morning someone on your team has a folder of PDFs, a shared drive named "final_final", and a question that sounds simple: what do we actually owe?

They search the way every team searches. They type the words they remember. `high-risk`. `penalty`. `Article 12`. The keyword engine is obedient. It returns every document that contains those tokens, ranked by how often the tokens appear. Forty hits. None of them is the clause that answers the question, because the clause says "providers of AI systems that pose a significant risk" and never repeats the shorthand the human typed. Or the opposite failure, which is worse: they typed `Article 12` exactly, and a semantic search box hands them Article 9, because both articles talk about obligations and the embedding thinks that is the point.

The code they need is in the corpus. The tool cannot think in both of the ways a legal question thinks. It matches words, or it matches a vibe. EU law asks for both, often in the same sentence.

That gap is not theoretical. It is the reason a compliance search demo looks finished and then embarrasses you in the first real week. Cloud assistants can paper over it, and then you have a second problem: the text of the law, plus your internal policy notes, leaving the building. For a lot of teams that is the end of the conversation.

So I built the search path I wanted to hand to that Friday morning. It runs against English EU legislation, it keeps the chunk text next to the vectors, and it answers with passages instead of a poem. Dense vectors for the question you could not quote. Learned sparse vectors for the citation you could. [Qdrant](https://qdrant.tech/) fuses the two ranked lists in one call. The project is [qdrant on GitHub](https://github.com/Anusha0501/qdrant). You can follow the modules while you read.

## Technical deep-dive: hybrid search paradigms in Qdrant

### The hybrid query is the whole design

Almost every "hybrid search" write-up still means the same architecture. A vector database sits next to Elasticsearch or a BM25 service. Your application embeds the query, calls both, and merges scores in Python with a constant you tuned on a laptop. It works until the two score scales drift, or someone adds a filter on one side and forgets the other.

For a corpus whose unit of meaning is a statutory clause, that split is a tax. You now operate two retrieval systems to solve one question.

Qdrant's hybrid query keeps both signals on the same point. You give the collection named vectors. One dense, one sparse. At query time you prefetch a candidate list from each, and you ask the server to fuse those lists with reciprocal rank fusion. Your process does not invent a formula that compares cosine to BM25. It sends one `query_points` request and reads one list back, payload included.

Here is the shape, taken from `src/search.py`:

```python
response = client.query_points(
    collection_name=settings.collection_name,
    prefetch=[
        qm.Prefetch(query=dense_vec, using="dense", limit=50),
        qm.Prefetch(
            query=qm.SparseVector(indices=indices, values=values),
            using="sparse",
            limit=50,
        ),
    ],
    query=qm.FusionQuery(fusion=qm.Fusion.RRF),
    limit=10,
    with_payload=True,
)
```

RRF cares about rank, not about the original score's unit. A passage that is near the top of both lists outranks a passage that won only one of them. That is the property you want when "the idea" and "the article number" disagree.

The sparse vector in the default path is not BM25. It is SPLADE, a learned sparse model. BM25 can only reward tokens the user typed. SPLADE still speaks in weighted tokens, so an article number can dominate the sparse hit, and it can also put weight on related terms the user did not type. I keep a BM25 vector on the same points only so a benchmark can compare the two hybrids fairly. The search command people run does not use it.

This "one point, two retrievers, fusion inside the database" choice is the decision the rest of the pipeline hangs off. If you remember one thing from the architecture, remember that.

**Figure 1.** Redraw this in Excalidraw before the doc is reviewed. Indexing on the left, query on the right. A legal act is split into chunks. Each chunk becomes a dense vector and a SPLADE vector, stored together with the text. A question walks the same two models, Qdrant prefetches, RRF returns the top chunks. The repo sketch is `diagrams/hybrid_search_architecture.svg`.

### What fails if you only ship one side

I use the same three probes whenever I change a parameter. They are the acceptance test, not a slogan.

| Query | Dense only | Dense + BM25 | Dense + SPLADE |
|---|---|---|---|
| What penalties must Member States lay down for infringement? | Usually fine. The idea is in the text. | Fine if the rare words were typed. | Fine, and more tolerant of "sanction" versus "penalty". |
| Article 12, transpose this Directive into national law | Often drifts to another "Member States shall". | Catches the number. Misses a paraphrase of "transpose". | Catches the number and can expand around the duty. |
| Powers of the Commission to adopt implementing acts | Partial. "Commission" is everywhere. | Exact tokens only. | Best of the three on this mix. |

Read the middle column carefully before you celebrate BM25. It is the right fix for a pure citation and a weak fix for legal language, where the operative verb and the verb in the user's mouth are cousins. SPLADE is the attempt to keep the citation behaviour without freezing the vocabulary.

**Figure 2.** One paragraph, two highlighters, drawn in Excalidraw. BM25 highlights only the query tokens. SPLADE highlights those and a small set of expanded legal terms. Do not draw it as magic. Draw it as a sparse vector that is allowed to be wider than the query string. Sketch in the repo: `diagrams/bm25_vs_splade.svg`.

### Architecture, in the order the code runs

When you run the README from top to bottom, five things happen.

1. `src/data_prep.py` downloads a capped sample of English EU law, chunks it on article boundaries, and writes `data/chunks.jsonl`. In the same run it writes `data/eval_queries.jsonl`, with gold chunk ids taken from those chunks.
2. `src/qdrant_setup.py` creates one collection with a dense named vector and two sparse named vectors.
3. `src/ingest.py` embeds every chunk and upserts points. Re-running uses a deterministic id, so you update in place.
4. `src/search.py` embeds the query and issues the hybrid query above.
5. `src/evaluate.py` runs the same questions three ways and prints Precision@K and mean reciprocal rank.

Indexing is the slow part, and you do it when the corpus changes. Querying is the part a person feels. The notebook at `notebooks/hybrid_search_demo.ipynb` runs the same five stages against an in-memory Qdrant so you can read the path without a server. Production points `QDRANT_URL` at Cloud or at a service you operate. The API key stays in the environment.

## Step-by-step hands-on implementation

Open `notebooks/hybrid_search_demo.ipynb` in Colab or locally. The first code cell clones [the repository](https://github.com/Anusha0501/qdrant) when `src/` is not already on disk. After that the notebook chunks a small LexGLUE sample, prints one dense vector and one SPLADE vector, creates the collection, ingests, runs the fused query, and prints the Precision@K / MRR table.

### The corpus has to be a dataset that still exists

The original first command loaded `joelniklaus/eurlex`. That repository is not on the Hugging Face Hub anymore. The call 404s, `data/chunks.jsonl` is never written, and ingest, search, and eval have nothing to do. A reviewer hit this immediately. Everything downstream was blocked.

The working corpus is LexGLUE's EUR-LEX config:

```python
from datasets import load_dataset

dataset = load_dataset(
    "coastalcph/lex_glue",
    "eurlex",
    split="train",
    revision="main",
)
```

This is English EU legislation, tens of thousands of documents in the train split, each with `text` and EuroVoc `labels`. It is a benchmark dataset from the LexGLUE paper, not a live feed of the Official Journal. Say that out loud in any demo. The EU AI Act is 2024 and is not in this slice. GDPR and the UK Data Protection Act are not downloaded by this command either. If you have those texts and you are allowed to index them, pass a directory:

```bash
python -m src.data_prep --max-docs 500 --local-dir ./local_acts --out data/chunks.jsonl
```

Files ending in `.txt` or `.md` are chunked with the same rules and appended. That is the supported way to put a newer regulation next to the LexGLUE background, not a pretend column in the Hub dataset.

Rows are normalized in one function so both shapes survive. LexGLUE has no CELEX id and no title. The document id becomes `eurlex-{row number}`, which stays stable for a given revision and a given cap. EuroVoc names, when the feature list exposes them, become the title printed next to a hit. An older row that still carries `celex_id` and `title` keeps them. Empty text is skipped. If you key documents off `row.get("id")` and the field does not exist, every id becomes empty and your eval labels collapse. That bug looks like "search is random". It is an id bug.

Pin the name in `.env` if you override defaults:

```text
HF_DATASET_NAME=coastalcph/lex_glue
HF_DATASET_CONFIG=eurlex
HF_DATASET_REVISION=main
```

If a previous `.env` still sets `HF_DATASET_NAME=joelniklaus/eurlex`, it will override the code default and the 404 comes back. Delete that line.

### Stage 1: chunk on the statute's own seams

The first mistake in legal retrieval is to treat an act like a blog post. You pick 512 tokens because a model card said so, and you slice. Article 12's duty ends up in two vectors, or it shares a vector with the start of Article 13. The embedding then represents a sentence nobody enacted.

`chunk_text` splits on boundaries first:

```python
ARTICLE_BOUNDARY_RE = re.compile(
    r"(?=\n\s*(Article\s+\d+|Art\.\s*\d+|Recital\s+\d+)\b)",
    re.IGNORECASE,
)
```

A segment shorter than 800 characters is one chunk. A longer article slides forward with 150 characters of overlap, so a cross-reference at the edge is still visible from both sides. The stored object is small and explicit:

```json
{
  "id": "eurlex-3::chunk0",
  "doc_id": "eurlex-3",
  "title": "approximation of laws",
  "chunk_index": 0,
  "text": "Article 12\nMember States shall bring into force..."
}
```

The id is the natural key. Ingest hashes it with UUID5, so a second ingest of the same chunk updates the point. You do not grow duplicates every time you fix a regex.

This parser will not impress a legislative drafter. Annexes, tables, and corrigenda are still messy. The property you are buying is narrower: the citation and the duty under it usually share a point. Hybrid search cannot recover a citation you sliced off the clause.

### Stage 2: one collection, three named vectors

`qdrant_setup.py` creates `eu_reg_directives` if it does not exist.

- `dense` uses cosine distance. The default model is `BAAI/bge-small-en-v1.5` through FastEmbed, 384 dimensions, local ONNX, no key. Set `EMBEDDING_PROVIDER=cohere` and `COHERE_API_KEY` for Cohere `embed-multilingual-v3.0`, which is 1024 dimensions. Collection setup reads the size from the model you configured. Recreate the collection when you change it. Do not hand-edit a constant and hope ingest agrees.
- `sparse` is SPLADE (`prithivida/Splade_PP_en_v1`). This is the learned sparse channel. The brief also names BGE-M3. This FastEmbed pin does not include BGE-M3 sparse weights, so the learned expansion model here is SPLADE. `Qdrant/bm42-all-minilm-l6-v2-attentions` is the other learned sparse model you can set with `SPARSE_MODEL_NAME`.
- `sparse_bm25` is FastEmbed's BM25 model (`Qdrant/bm25`). The evaluator uses it. Default search does not.

A keyword payload index on `doc_id` is created with the collection. You will want more fields later, year, instrument type, in force or not. Add them when you have a source for them. Do not block the first index on a metadata model you have not got.

```bash
python -m src.qdrant_setup
```

### Stage 3: embed, then upsert the text with the vectors

Ingest reads `data/chunks.jsonl` in batches of 64. Each batch is embedded as documents, not as queries. That distinction matters for Cohere, which wants `search_document` at index time and `search_query` at query time. FastEmbed's default path does not make that split, and the code still threads `is_query` through so the Cohere branch stays honest.

Each point stores all three vectors and a payload: chunk id, document id, title, chunk index, and the full chunk text. Storing the text in the payload means a hit can be shown without a second lookup. The index is the retrieval artifact. When a lawyer asks "show me the sentence", you have the sentence.

```bash
python -m src.ingest --chunks data/chunks.jsonl
```

If the collection was never created, ingest raises and tells you to run setup. I would rather stop than write into a collection with the wrong vector names and discover it at query time.

### Stage 4: ask, and show the passage

```bash
python -m src.search \
  "What penalties must Member States lay down for infringement of this Regulation?"
```

The CLI prints rank, score, document id, title, and the opening of the chunk. On LexGLUE the title is often a EuroVoc label such as environment, fisheries, or competition. Use it as a sanity check, not as a legal classification you trust blindly. Penalties language shows up in acts whose EuroVoc label is not "criminal law". Read the text.

Dense-only is the same function with `--mode dense`. Dense plus BM25 is `--mode dense_bm25`. You should not need those in a product path. They exist so the next stage is a comparison, not a feeling.

Query embeddings and document embeddings come from the same model family. Mixing a Cohere query vector with a FastEmbed document vector will produce scores that look like numbers and mean nothing. The settings object is the single place that choice is made. Do not construct a second client in a notebook "just to try a model" against the old collection.

## Evaluation and quality benchmarks

`python -m src.evaluate --queries data/eval_queries.jsonl` used to die immediately. The file was not in the repository. Gitignore had `data/` as a directory rule, and a later negation for `data/eval_queries.jsonl` cannot bring a file back once the parent directory is ignored. The metrics code was fine. The input was absent. Precision@K and MRR never printed, so any claim that SPLADE hybrid wins was unmeasured.

Data prep now writes the file. Each line is one JSON object:

```json
{"query": "How must Member States transpose this Directive into national law?", "relevant_chunk_ids": ["eurlex-3::chunk0"]}
```

Curated questions are included only when their anchor phrases occur in the chunks you actually built. A penalties question with no penalties clause in the sample is dropped, instead of being scored against an empty gold set. The remaining lines are known-item questions: one per document, built from an Article or Recital mention plus a few content words, with that chunk id as the only relevant hit.

If you already chunked and you only need labels:

```bash
python -m src.data_prep --from-chunks data/chunks.jsonl --eval-out data/eval_queries.jsonl
python -m src.evaluate --queries data/eval_queries.jsonl --k 5
```

The report is three rows. Dense-only. Dense plus BM25. Dense plus SPLADE. Precision at 5, and MRR.

A note on how to read it, because this setup will otherwise start a useless argument. Known-item queries have a single gold id. If the right chunk is rank 1, reciprocal rank is 1, and Precision@5 is 0.2, because the other four results are counted as misses. MRR is the headline for that query shape. Precision@K is easier to read on the curated questions, which may carry up to three gold chunks. Put that sentence under the table in the doc. Do not publish a number from a different run, and do not fill the table with a guess. Run the command on the index you are willing to talk about, and paste that table.

What the harness is allowed to support, qualitatively, is the table in the earlier section. Dense drifts on citations. BM25 hybrid is literal. SPLADE hybrid is the default because the Friday question will not tell you which failure it is about to be.

**Figure 3.** Make this the clip. Same query, two panes. Left, dense-only, wrong article, high score, lots of "shall". Right, hybrid, the article number the user typed, the duty underneath it. Six seconds of terminal or a short Excalidraw animation. A static architecture diagram will not travel on a social feed. This one can.

### What you can swap, and what you should not pretend

The stack is small on purpose.

- Qdrant, Cloud or local, for storage and fusion. The notebook's `:memory:` client is a reading aid. It is not the deployment.
- FastEmbed for SPLADE, for BM25, and for the default dense model. CPU is enough to start.
- Cohere, optional, for multilingual dense embeddings. It does not replace the sparse model.
- The `datasets` library, to pull LexGLUE.

You can swap the dense model. Recreate the collection when the dimension changes. You can add a local directory of acts you are allowed to hold. You can raise `--max-docs` once a 500-document run has taught you what the chunks look like. The full LexGLUE train split is the right eventual corpus and a bad first afternoon.

You should not swap in a generative model and call the passage list a legal answer. The payload is there so a later reader, human or model, can quote a real sentence. A paragraph that names the wrong article with confidence is a defect, not a feature. If you add that layer, constrain it to the retrieved text, and show the chunk beside the prose.

You should not describe this index as the current consolidated law. EUR-Lex is the official source. LexGLUE is a frozen research cut. In-force status, amendments, and corrigenda are product requirements, and they are metadata you still have to model. Retrieval quality does not substitute for currency.

A few limits I would put in the same paragraph as any demo video.

Character windows are not token windows. Overlap can surface the same sentence twice. Deduplicate in the UI by document and article if that annoys people.

English sparse models against an English LexGLUE config is a matched pair. A multilingual dump needs a different sparse plan. Do not discover that in production.

Filters are thin. `doc_id` is indexed. A real tool filters by year and by instrument. The point payload can grow. The fusion call already accepts Qdrant filters when you are ready to pass them.

The eval set is a harness, not a certification. Replace the curated questions with questions your users sent, and have a lawyer mark the passages. Keep the file format. The scorer does not care how the gold ids were chosen.

### Run it without skipping a step

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `QDRANT_URL`. Then, in order:

```bash
python -m src.data_prep --max-docs 500 --out data/chunks.jsonl
python -m src.qdrant_setup
python -m src.ingest --chunks data/chunks.jsonl
python -m src.search "What penalties must Member States lay down for infringement of this Regulation?"
python -m src.evaluate --queries data/eval_queries.jsonl
```

Step 1 is what makes step 5 possible. It writes the chunks and the gold file together. The checked-in `data/eval_queries.jsonl` shows the schema. It is overwritten so the ids match the chunks you just built. Evaluating the schema file against some older collection will score zero. That zero is an alignment error. Fix it with `--from-chunks`, not with a new fusion algorithm.

Tests that do not need the network cover chunk boundaries, the LexGLUE row mapping, the eval-file alignment, Precision@K, MRR, and deterministic point ids. `pytest` is enough to see that the pure logic holds before you spend the download.

### The part worth keeping

Keyword search and dense search fail on different Thursdays, and legal work has both Thursdays. The repair is not a second database and a Python merge you will stop trusting. It is a Qdrant point that carries a dense vector and a learned sparse vector, a prefetch on each, and RRF so rank is the common language.

The implementation details that made this runnable are smaller than the architecture and just as necessary. Load `coastalcph/lex_glue` with config `eurlex`, because `joelniklaus/eurlex` is a 404. Write `data/eval_queries.jsonl` from the same chunks you index, because a benchmark without gold ids does not run. After that, the interesting work is the work the code cannot do for you: which official text you stand behind, and which questions a person actually asked.

## References

- Qdrant, hybrid queries and RRF: https://qdrant.tech/documentation/concepts/hybrid-queries/
- Qdrant, query API: https://qdrant.tech/documentation/concepts/search/
- Qdrant FastEmbed, dense, sparse, and BM25: https://qdrant.tech/documentation/fastembed/
- Formal, Piwowarski, and Clinchant, SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking: https://arxiv.org/abs/2109.10086
- LexGLUE on Hugging Face, EUR-LEX config: https://huggingface.co/datasets/coastalcph/lex_glue
- Chalkidis, Jana, Hartung, Bommarito, Androutsopoulos, Katz, and Aletras, LexGLUE: A Benchmark Dataset for Legal Language Understanding in English, ACL 2022: https://aclanthology.org/2022.acl-long.297/
- EUR-Lex: https://eur-lex.europa.eu/
- Regulation (EU) 2024/1689, the EU AI Act: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- Regulation (EU) 2016/679, GDPR: https://eur-lex.europa.eu/eli/reg/2016/679/oj
- Data Protection Act 2018 (UK): https://www.legislation.gov.uk/ukpga/2018/12/contents

---

## Cuts for LinkedIn, Hashnode, and X

Post on a weekday morning. Leave 24–48 hours before the next post. Tag Qdrant, and tag the Medium publication if one of them takes the piece. The article should stay free to read. The long-term value is search, not a metered view.

### LinkedIn

Keyword search on EU law returned every PDF that contained the word I typed. Semantic search returned a neighbouring article and sounded sure of itself.

The clause I needed was already in the corpus. One tool could see the words and not the duty. The other could see the duty and not the article number. Legal questions do both, sometimes in one line.

The index I ended up with puts a dense vector and a SPLADE sparse vector on the same Qdrant point. One hybrid query prefetches both lists and fuses them with reciprocal rank fusion. The chunk text comes back with the hit. There is no second search engine to keep in sync.

Two fixes were the difference between a README and a run:

- `joelniklaus/eurlex` 404s on Hugging Face. The loader now uses LexGLUE, `coastalcph/lex_glue`, config `eurlex`.
- The Precision@K file was missing, so the benchmark could not start. Data prep writes `data/eval_queries.jsonl` with gold ids from that same chunk file.

Repo: https://github.com/Anusha0501/qdrant
Essay: https://MEDIUM_URL

What is the query that made your dense-only legal search look finished and then fail? Drop it below. I want the wording, not the stack.

### Hashnode

Publish the article above, and place this directly under the title:

First published on Medium: https://MEDIUM_URL

### X

1. Keyword search gave me every EU act that contained the token. Dense search gave me a neighbouring article and a high score. The clause I wanted was already indexed.
2. A citation and a paraphrase show up on the same afternoon. Legal search that only does one of them is a demo.
3. Same Qdrant point: dense vector for intent, SPLADE sparse vector for the statutory words. Prefetch both. Fuse with RRF. BM25 stays in the benchmark.
4. Practical repairs: the old Hub dataset 404s, so this loads LexGLUE eurlex. And the eval file is generated from the chunks, so Precision@K and MRR can actually run.
5. https://github.com/Anusha0501/qdrant
   https://MEDIUM_URL
