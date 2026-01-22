from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from backend.qdrant_utils import QdrantHandler

class RagPipeline:
    def __init__(self):
        self.qdrant = QdrantHandler()
        self.flash_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./opt")

    def bm25_search(self, query: str, documents: list[dict], k=5):
        if not documents:
            return []
        
        # Simple tokenization for POC
        tokenized_corpus = [
            (doc['title'] + " " + doc.get('description', '')).lower().split() 
            for doc in documents
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        
        scores = bm25.get_scores(tokenized_query)
        # Zip docs with scores and sort
        docs_with_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in docs_with_scores[:k]]

    def reciprocal_rank_fusion(self, list_1, list_2, k=60):
        # A simplified RRF implementation
        fusion_scores = {}
        
        for rank, doc in enumerate(list_1):
            doc_id = doc['book_id']
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = {'doc': doc, 'score': 0.0}
            fusion_scores[doc_id]['score'] += 1 / (k + rank + 1)
            
        for rank, doc in enumerate(list_2):
            doc_id = doc['book_id']
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = {'doc': doc, 'score': 0.0}
            fusion_scores[doc_id]['score'] += 1 / (k + rank + 1)
            
        sorted_results = sorted(fusion_scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_results]

    def run_pipeline(self, query: str, top_k=5):
        # 1. Dense Search (Qdrant)
        dense_results = self.qdrant.search_dense(query, limit=10)
        
        # 2. BM25 Search (Recovers all docs from "cache" for POC BM25)
        # Ideally, BM25 index is persistent. Here we rebuild/scan for POC.
        all_docs = self.qdrant.get_all_documents_for_bm25()
        bm25_results = self.bm25_search(query, all_docs, k=10)
        
        # 3. Metadata Filtering (Implicit in Qdrant search if added, skipping strict step here for brevity)

        # 4. RRF Fusion
        fused_results = self.reciprocal_rank_fusion(dense_results, bm25_results)
        
        # 5. Flash Reranker
        if not fused_results:
            return []
            
        passages = [
            {"id": doc['book_id'], "text": f"{doc['title']} {doc.get('description','')}"} 
            for doc in fused_results
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.flash_ranker.rerank(rerank_request)
        
        # Map back to full doc objects
        final_docs = []
        for r in reranked_results[:top_k]:
            # find original doc
            original = next((d for d in fused_results if d['book_id'] == r['id']), None)
            if original:
                # Inject score can be useful
                original['_rerank_score'] = r['score']
                final_docs.append(original)
                
        return final_docs