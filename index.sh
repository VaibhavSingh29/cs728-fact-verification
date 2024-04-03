python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ./data/processed_wiki_ner/ \
  --index indexes/wiki-pages-ner \
  --fields tag \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw