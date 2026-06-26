# AssemblyBot Search Demo

This directory contains two small ways to show semantic search over the included
SQLite database.

## Quick demo: `search_demo.py`

Use this when you want the fastest proof that semantic search works.

```bash
python demo/search_demo.py
```

`search_demo.py` uses precomputed query vectors from `demo_queries_fand_flags.py`,
so it does not need `numpy`, `sentence-transformers`, or model downloads. It only
uses the Python standard library and the included `assemblybot.sqlite` file.

## Full search: `search.py`

Use this when you want to type a natural-language query and embed it at runtime.
The bundled model is French, and the transcript text in the database is French
too, so queries should be written in French for sensible results.

```bash
pip install -r demo/requirements-search.txt
python demo/search.py "qui a parle de la jeunesse"
```

`search.py` loads a SentenceTransformer model, embeds your query, searches the
semantic chunks in `assemblybot.sqlite`, and lets you open a returned `turn_id`
to read the full turn and its quality flags.

This path needs dependencies because the query embedding is computed live.
