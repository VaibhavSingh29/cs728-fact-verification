python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ./data/processed_wiki/ \
  --index indexes/wiki-pages \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw