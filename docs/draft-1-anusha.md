<!--
Paste everything under the title into the Google Doc.
Redraw Figure 1, Figure 2, and Figure 3 in Excalidraw (2–3 images).
Do not paste the "Cuts for LinkedIn, Hashnode, and X" section into Medium.
Target length of the article body: about 2200–3500 words, plus the code blocks.
-->

# The Clause Was in the Corpus. Dense Search Walked Past It.

## Introduction and the LegalTech challenge

I can still see the query. It was not poetic. It was the kind of line a lawyer types when they are already annoyed: `Article 12`, then something about transposing a directive into national law. The passage was in the index. I had chunked it myself. Dense search came back with a confident neighbour: another article, another duty, the same fog of "Member States shall". The citation I had typed was sitting a few hundred characters away, and the model had walked past it because the embedding had decided the *topic* was close enough.

That is the whole problem this project is about. EU legislation does not fail in one way. Sometimes the person searching knows the idea and not the number. Sometimes they know the number and will not accept a paraphrase. A search stack that only does one of those jobs will look brilliant in a demo and careless the first week a compliance team uses it.

So I built a hybrid index on [Qdrant](https://qdrant.tech/) for English EU law. Dense vectors carry the intent. Learned sparse vectors, SPLADE, carry the statutory grip: article numbers, "implementing acts", "enter into force", the words that are not allowed to be approximately right. Reciprocal rank fusion merges the two lists inside one Qdrant query. The code is in [github.com/Anusha0501/qdrant](https://github.com/Anusha0501/qdrant). This piece is the walkthrough I wish I had written before the first review, including the two things that actually blocked a clean run.

**Figure 1.** Draw this in Excalidraw. Left side, indexing: an EU act is split on Article and Recital boundaries, then each chunk is embedded twice, once with a dense model (`BAAI/bge-small-en-v1.5`, 384 dimensions, cosine) and once with SPLADE. Both vectors land in one Qdrant point, with the chunk text in the payload. Right side, query: the same two embedders run on the question, Qdrant prefetches a candidate list from each named vector, and RRF returns one ranked list. A sketch of this layout is already in the repo at `diagrams/hybrid_search_architecture.svg`.

## Technical deep-dive: hybrid search paradigms in Qdrant

### Three queries, and only one of them is semantic

I keep three queries on a sticky note when I test legal search. They look similar. They are not.

The first is intent. "What penalties must Member States lay down for infringement of this Regulation?" A person can ask that without remembering Article 3. A dense model is built for this. It should find the penalties clause even when the wording drifts.

The second is a citation. "How must Member States transpose this Directive into national law?" or, sharper, "Article 12" plus "transpose". Here the number and the verb are the search. If you return a different article that is *about* Member State duties, you have failed, even if the cosine score looks healthy.

The third is both at once. "Which powers does the Commission have to adopt delegated or implementing acts?" You need the role (the Commission), the instrument type (delegated or implementing acts), and the legal flavour of the sentence. Keyword search gets the rare phrase and misses the paraphrase. Dense search gets the flavour and drops the phrase.

Classic BM25 hybrid, dense plus a lexical score, fixes the second query and stays brittle on the third. BM25 can only reward tokens that were actually typed. SPLADE is a sparse neural model: it still emits a bag of weighted tokens, so an article number can dominate, but it has learned which tokens in a passage matter and it can expand the query toward related legal words. That expansion is the difference I cared about. I did not want a second index and a homemade merger. I wanted both signals in one collection, fused by the database.

**Figure 2.** Draw a single chunk in Excalidraw, two highlighters. Top highlighter, BM25: only the words that appear in the query are lit. Bottom highlighter, SPLADE: those words are lit, and so are a few neighbours the model believes belong to the same legal idea ("sanction" next to "penalty", "implementing act" next to "delegated"). Caption it honestly. SPLADE is not a thesaurus you maintain. It is a sparse vector. The sketch in `diagrams/bm25_vs_splade.svg` is the same idea.

## Step-by-step hands-on implementation

The notebook `notebooks/hybrid_search_demo.ipynb` is the Colab path. Its first code cell clones the repo when `src/` is missing, then the cells chunk, embed one passage with dense and SPLADE, create the multi-vector collection, ingest, run the RRF query, and print Precision@K and MRR.

### Dataset preparation

The first command in the README used to be:

```bash
python -m src.data_prep --max-docs 500 --out data/chunks.jsonl
```

It loaded `joelniklaus/eurlex` from Hugging Face. That repository is gone. The Hub returns 404, and every later step is blocked because there are no chunks to embed. I switched the default to LexGLUE, which is maintained, documented, and boring in the way you want a corpus to be boring.

```python
from datasets import load_dataset

dataset = load_dataset("coastalcph/lex_glue", "eurlex", split="train")
```

LexGLUE's `eurlex` config is English EU legislation published on EUR-Lex, annotated with EuroVoc concepts. The train split is large, on the order of 55,000 acts, with a validation split and a test split beside it. Each row is a `text` field and a `labels` field. There is no CELEX identifier and no title in the Hub schema. If you pretend those columns exist, every document id collapses and your eval file points at nothing.

The loader in `src/data_prep.py` accepts both shapes. A LexGLUE row becomes a stable id `eurlex-{row index}`, and the EuroVoc class names, when the dataset exposes them, become the title you see in search results. An older dump that still has `celex_id`, `title`, and `celex_text` still loads. Local `.txt` or `.md` files passed with `--local-dir` are appended after the Hub sample. That is how GDPR, the UK Data Protection Act, or the EU AI Act enter this index. They are not inside LexGLUE. The AI Act is 2024. LexGLUE's EUR-LEX slice comes from earlier public releases. If a demo promises the AI Act and only downloads LexGLUE, the demo is lying. I would rather say: the default corpus is EU legislation with EuroVoc labels, and anything newer is a file you supply because you have the right to use it.

Environment variables hold the dataset pin, same as the Qdrant URL and the API key. Nothing secret sits in the repository.

```text
HF_DATASET_NAME=coastalcph/lex_glue
HF_DATASET_CONFIG=eurlex
HF_DATASET_REVISION=main
```

`revision` stays explicit so a re-run does not silently pick up a different Hub commit.

### Chunk on the article, not on a round number of tokens

Legal text is a bad citizen of sliding windows. An obligation often starts at "Article 12" and runs until the next article. If you cut at 512 tokens because that is what the model card suggests, you store half a duty and half of the next one. The embedding then represents a sentence that no legislator wrote.

I split on article and recital boundaries first:

```python
ARTICLE_BOUNDARY_RE = re.compile(
    r"(?=\n\s*(Article\s+\d+|Art\.\s*\d+|Recital\s+\d+)\b)",
    re.IGNORECASE,
)
```

A segment that already fits in the chunk size, 800 characters by default, stays whole. A longer article is windowed with a 150-character overlap so a cross-reference near the cut is not orphaned. Empty text is dropped. Each stored chunk carries `id`, `doc_id`, `title`, `chunk_index`, and `text`. The id looks like `eurlex-12::chunk0`. Ingest turns that string into a UUID5, so a second run upserts the same point instead of duplicating it.

This is not a perfect legal parser. Nested paragraphs, annexes, and corrigenda will still surprise you. It is good enough that a citation and the duty it introduces tend to land in the same point, which is the property hybrid search actually needs.

### One collection, named vectors, no second database

Qdrant can store several vectors on one point. I use that instead of running a vector database beside Elasticsearch and hoping my fusion code agrees with itself.

The collection `eu_reg_directives` has:

- `dense`: cosine vectors from FastEmbed. The default model is `BAAI/bge-small-en-v1.5`, 384 dimensions. Set `EMBEDDING_PROVIDER=cohere` and `COHERE_API_KEY` to use Cohere `embed-multilingual-v3.0`, which is 1024 dimensions. `qdrant_setup` reads that size when it creates the collection. The sparse side stays FastEmbed either way. This FastEmbed pin ships SPLADE as the learned sparse model. It does not ship BGE-M3 sparse weights. If you want a lighter learned sparse model, set `SPARSE_MODEL_NAME=Qdrant/bm42-all-minilm-l6-v2-attentions` and recreate the collection.
- `sparse`: SPLADE, model `prithivida/Splade_PP_en_v1`. This is the learned sparse vector the default search uses.
- `sparse_bm25`: Qdrant FastEmbed's BM25 sparse model. It exists so the benchmark can compare Dense+BM25 against Dense+SPLADE on the same points. The default query does not read it.

Payload fields are the chunk id, document id, title, chunk index, and the raw text. `doc_id` has a keyword payload index so a later filter, "only this CELEX, only this file", does not scan the whole collection.

Create it with:

```bash
python -m src.qdrant_setup
```

Point `QDRANT_URL` at Qdrant Cloud or at a local server. For a notebook, an in-memory client is enough; the demo notebook does that so you can read the fusion path without standing up Docker. Production is the URL and the API key. The key belongs in the environment. `.env` is gitignored.

Ingest is a batch loop. Each batch is embedded three times, dense, SPLADE, and BM25, and upserted. If the collection is missing, ingest stops and tells you to run setup. I would rather fail there than create a half-configured collection from a write path.

### The query is one round trip

`src/search.py` is the part I want a reader to actually open. Dense-only is a single `query_points` against the `dense` vector. Hybrid and Dense+BM25 share a shape: two prefetches, then a fusion query.

```python
response = client.query_points(
    collection_name=settings.collection_name,
    prefetch=[
        qm.Prefetch(query=dense_vec, using="dense", limit=50),
        qm.Prefetch(
            query=qm.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
            using="sparse",  # or "sparse_bm25" in the benchmark mode
            limit=50,
        ),
    ],
    query=qm.FusionQuery(fusion=qm.Fusion.RRF),
    limit=10,
    with_payload=True,
)
```

Prefetch asks each named vector for its own top 50. RRF, reciprocal rank fusion, does not care that one score is cosine and the other is a sparse dot product. It cares about rank. A chunk that is third in both lists beats a chunk that is first in one list and absent from the other. That is the behaviour I want when a citation and a paraphrase disagree about who should win.

I kept the prefetch limit at 50 and the final limit at 10 as defaults. Those are knobs, not doctrine. If your corpus grows into the full 55,000 acts and you chunk aggressively, raise the prefetch before you reach for a reranker. A reranker on the wrong 10 candidates cannot see the article you dropped.

Run it:

```bash
python -m src.search \
  "What penalties must Member States lay down for infringement of this Regulation?"
```

The CLI prints score, document id, title, and the first 280 characters. The title, on LexGLUE, is the EuroVoc labels. That is a useful smell test. If a penalties question returns a fisheries label and a fisheries paragraph, read the paragraph before you blame fusion. Sometimes the gold text really is in a fisheries regulation, because penalties language is everywhere in EU law. The benchmark exists so you do not settle that argument by vibes.

## Evaluation and quality benchmarks

The last command used to be:

```bash
python -m src.evaluate --queries data/eval_queries.jsonl
```

The repository gitignored `data/` in a way that also hid `data/eval_queries.jsonl`, and the file was never generated. Python raised `FileNotFoundError`. Precision@K and mean reciprocal rank never ran, so there was no evidence for the claim that SPLADE hybrid beats BM25 hybrid. A table you cannot reproduce is a slogan.

Two changes make the command real.

First, gitignore now ignores the contents of `data/` and then un-ignores `data/eval_queries.jsonl`. A parent directory rule of `data/` cannot be undone for one child file. That was the mechanical bug.

Second, `data_prep` writes the query file itself, with gold chunk ids taken from the chunks it just produced. If you already have `data/chunks.jsonl` and you only need labels:

```bash
python -m src.data_prep --from-chunks data/chunks.jsonl --eval-out data/eval_queries.jsonl
python -m src.evaluate --queries data/eval_queries.jsonl
```

Each line looks like this:

```json
{"query": "How must Member States transpose this Directive into national law?", "relevant_chunk_ids": ["eurlex-3::chunk0"]}
```

The builder does two jobs. It keeps a short list of curated compliance questions, and it keeps a question only when the anchors actually occur in this chunk set. "Penalties" is not a gold label if this sample of acts never mentions penalties. Then it fills the rest with known-item questions, one per document, derived from an Article mention and a few content words, with that chunk id as the single relevant hit. The copy of the file in git is a schema example. Step 1 of the README overwrites it so the ids match that run. If you evaluate the checked-in file against a different index, every score will be zero, and the zero will be your fault, not RRF's.

The metrics are the plain ones. Precision at K is the fraction of the top K ids that sit in the gold set. Reciprocal rank is `1 / rank` of the first gold hit, or zero. The harness runs the same queries in three modes: `dense`, `dense_bm25`, and `hybrid`. Print the table. Do not publish a table from a different corpus and hope nobody diffs the code.

I am not going to invent a Precision@K number in this article. The score depends on how many documents you indexed, which queries survived the anchor filter, and whether gold is a single chunk. Known-item retrieval with one relevant id makes MRR the number to read, and it keeps Precision@5 modest even when the right chunk is at rank 1, because the other four slots count as misses. Say that next to the table or the table will be misread.

What I will claim, because it is the reason the three modes exist:

- Dense-only is the right tool when the user describes a duty and the lexical form varies.
- Dense plus BM25 is the right correction when the user typed a token that must appear, and the wrong correction when the important word is a legal cousin rather than a string match.
- Dense plus SPLADE is the default because citation queries and intent queries show up in the same afternoon, and I do not want the caller to pick a mode.

**Figure 3.** This is the one that should move. Record the terminal, or animate two result lists in Excalidraw. Same query, left column dense-only, right column hybrid. The left list's first hit is thematically related and cites the wrong article. The right list's first hit contains the article number the user typed and the operative sentence under it. Six to eight seconds is enough. Social posts die without motion. A still screenshot of a JSON line will not do the job this figure is for.

### What I would not tell a lawyer yet

This index retrieves passages. It does not decide what the law is. EU law is amended, consolidated, and sometimes corrected. LexGLUE is a research corpus with a publication date, not the EUR-Lex cellar at the minute you search. If a product answer is "Article 12 requires X", the product also needs the consolidation date and a link back to the official journal. I store the chunk text so the model, if you add one later, has something to quote. I have not added a generative layer here on purpose. A fluent paragraph that cites the wrong article is worse than a ranked list that shows the text.

A few engineering limits, said plainly.

The dense size is no longer a hardcoded 384. Collection setup asks FastEmbed for the configured model's dimension, and uses 1024 when the provider is Cohere. Switch models only on an empty collection. An existing collection keeps the size it was created with.

Chunking is character-based after the article split. It is not a tokenizer. Overlap of 150 characters can duplicate a sentence into two points, and both can return. Deduplicate by `doc_id` plus article mention in the application if the UI looks stuttery.

SPLADE and BM25 through FastEmbed are English-leaning. LexGLUE EUR-LEX in this config is English. The moment you index an authentic multilingual EUR-Lex dump, the sparse models need a second look, and Cohere's multilingual dense model becomes the interesting dense option. Do not flip the provider and keep the old collection.

Filters are barely used. A real compliance tool filters by instrument type, year, in-force status, and celex. The payload has `doc_id` and `title` so you can start. It does not have a proper metadata model. Add fields when you have them. Do not stuff the entire EUR-Lex notice XML into the vector and hope.

The benchmark's curated questions are anchors, not a legal test suite. They tell you whether retrieval can find a penalties clause, an entry-into-force clause, a transposition clause. They do not tell you whether your product is safe to put in front of a regulated customer. Build that set with a lawyer, from questions your users already asked, with gold passages they marked. Then keep the harness. The code path does not need to change.

### Run it in order

From a clean checkout:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `QDRANT_URL`. Leave `QDRANT_API_KEY` empty for a local server without auth. Then:

```bash
python -m src.data_prep --max-docs 500 --out data/chunks.jsonl
python -m src.qdrant_setup
python -m src.ingest --chunks data/chunks.jsonl
python -m src.search "What penalties must Member States lay down for infringement of this Regulation?"
python -m src.evaluate --queries data/eval_queries.jsonl
```

`--max-docs 500` is a deliberate cap. The full train split is a real download and a real embed. Start smaller if you are on a laptop. The notebook `notebooks/hybrid_search_demo.ipynb` does the same path with 25 documents and an in-memory Qdrant, and it writes the eval file in the same cell as the chunks.

If the first command fails on a dataset name, check that `.env` does not still say `HF_DATASET_NAME=joelniklaus/eurlex`. An old env file overrides the new default. Delete that line or set it to `coastalcph/lex_glue` with `HF_DATASET_CONFIG=eurlex`.

### What I want you to take from this

Hybrid search is not "vectors, but also keywords" as a slogan. It is a decision about which failures you refuse to ship. I refuse to ship a legal search that cannot hold onto an article number. I also refuse to ship one that only lights up when the user quotes the statute back to itself. Qdrant's prefetch and RRF let that decision live in one query, on one point, with the text attached so a human can see why a hit won.

The corpus has to be a corpus that exists. `joelniklaus/eurlex` does not. `coastalcph/lex_glue` with config `eurlex` does, and a review run of this pipeline against it returns real EU-law hits. The benchmark has to be a file that exists, with chunk ids from the same run as the index. Both of those are fixed in the repository now. The remaining work is the work that was always going to take a human: which questions your users ask, which passages a lawyer marks, and which official text you are willing to stand behind when the ranked list is no longer a demo.

## References

- Qdrant hybrid queries and Reciprocal Rank Fusion: https://qdrant.tech/documentation/concepts/hybrid-queries/
- Qdrant query API: https://qdrant.tech/documentation/concepts/search/
- Qdrant FastEmbed, including sparse and BM25 models: https://qdrant.tech/documentation/fastembed/
- SPLADE, Formal et al.: https://arxiv.org/abs/2109.10086
- LexGLUE dataset: https://huggingface.co/datasets/coastalcph/lex_glue
- Chalkidis et al., LexGLUE: A Benchmark Dataset for Legal Language Understanding in English, ACL 2022: https://aclanthology.org/2022.acl-long.297/
- EUR-Lex, the official source of EU law: https://eur-lex.europa.eu/
- EU AI Act, Regulation (EU) 2024/1689: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- GDPR, Regulation (EU) 2016/679: https://eur-lex.europa.eu/eli/reg/2016/679/oj
- UK Data Protection Act 2018: https://www.legislation.gov.uk/ukpga/2018/12/contents

---

## Cuts for LinkedIn, Hashnode, and X

These are not part of the Medium article. Post the article on a weekday morning. Wait 24–48 hours before the next post. Tag Qdrant. If a Medium publication picks it up, tag that publication too.

### LinkedIn

I typed an article number into a vector search over EU law and got back a different article with the same vibe.

The clause was in the index. Dense search walked past it, because "Member States shall" looks like "Member States shall" everywhere. Keyword search would have caught the number and missed the question I ask when I do not remember the number.

So the index stores both. A dense vector for intent, a SPLADE sparse vector for the statutory words, and Qdrant fuses the two lists with reciprocal rank fusion in one query. No sidecar Elasticsearch. The text of the chunk comes back with the hit, so a person can see the sentence that won.

Two practical notes from making the repo actually run:

1. `joelniklaus/eurlex` on Hugging Face returns 404. The pipeline now loads LexGLUE, `coastalcph/lex_glue`, config `eurlex`.
2. Precision@K never ran because `data/eval_queries.jsonl` was missing. Data prep writes that file now, with gold chunk ids from the same run as the index.

Code: https://github.com/Anusha0501/qdrant

Article: https://MEDIUM_URL

If you have shipped search for contracts, statutes, or policies, I want the query that embarrassed your dense-only demo. I will read them.

### Hashnode

Use the article above unchanged, and put this line under the title:

First published on Medium: https://MEDIUM_URL

### X

1. Dense search over EU law handed me a neighbouring article and called it a match. The citation I typed was in the index. The embedding walked past it.
2. Legal queries come in two shapes on the same afternoon: "what are the penalties" and "Article 12". One index has to survive both.
3. I store a dense vector and a SPLADE sparse vector on the same Qdrant point, and fuse them with RRF. BM25 is in the benchmark, not in the default query.
4. The Hub dataset `joelniklaus/eurlex` is a 404. LexGLUE's eurlex config is the corpus that actually downloads. Eval queries are generated from the chunks, so Precision@K has gold ids.
5. Code: https://github.com/Anusha0501/qdrant
   Write-up: https://MEDIUM_URL
