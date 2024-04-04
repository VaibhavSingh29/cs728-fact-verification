from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyserini.search.lucene import LuceneSearcher, querybuilder
from typing import List
class SparseRetriever(LuceneSearcher):
    def __init__(self) -> None:
        super().__init__('./save/indexes/wiki-pages-ner')
        self.searcher = LuceneSearcher('./save/indexes/wiki-pages-ner')
        self.stop_words = set(stopwords.words('english'))

    def prune_query(self, query: str):
        words = word_tokenize(query)
        filtered_words = [word for word in words if word not in self.stop_words]
        return filtered_words

    def generate_query(self, query: str, tags: List[str]):
        terms = self.prune_query(query)

        should = querybuilder.JBooleanClauseOccur['should'].value
        bool_query_builder = querybuilder.get_boolean_query_builder()

        for tag in tags:
            try:
                qtag = querybuilder.get_term_query(tag, field='tag')
                # boost = querybuilder.get_boost_query(qtag, 2.0)
                bool_query_builder.add(qtag, should)
                # bool_query_builder.add(boost, should)
            except:
                pass # tag is not in index
        
        for term in terms:
            try:
                qterm = querybuilder.get_term_query(term)
                bool_query_builder.add(qterm, should)
            except:
                pass # term is not in index
        
        query = bool_query_builder.build()
        return query

    def search(self, query: str, tag: List[str], k: int = 5):
        bool_query = self.generate_query(query, tag)
        hits = self.searcher.search(bool_query, k=k)
        if len(hits) < k:
            extra_docs = self.searcher.search(query, k=k-len(hits))
            hits.extend(extra_docs)
        return hits

    def batch_search(self, queries: List[str], tags: List[str], qids: List[str], k: int = 5):
        batch_hits = {}
        for i in range(len(queries)):
            hits = self.search(queries[i], tags[i], k=k)
            batch_hits[qids[i]] = hits
        return batch_hits