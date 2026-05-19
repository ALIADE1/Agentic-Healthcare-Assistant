import os
import json
import numpy as np
from typing import Dict, Any, Optional, List

class SemanticCache:
    """
    A semantic cache that stores queries and their corresponding responses.
    Uses sentence-transformers to compute embeddings and performs cosine similarity
    for semantic lookup. Fallbacks to keyword matching if the transformer model is unavailable.
    """
    def __init__(self, cache_file: str = "core/semantic_cache.json", threshold: float = 0.88):
        self.cache_file = os.path.abspath(cache_file)
        self.threshold = threshold
        self.cache_data = []
        self.model = None
        self.load_cache()
        self.init_model()

    def load_cache(self):
        """Loads cached responses from a persistent JSON file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache_data = json.load(f)
            except Exception as e:
                print(f"Error loading semantic cache file: {e}. Starting fresh.")
                self.cache_data = []
        else:
            self.cache_data = []

    def save_cache(self):
        """Persists the cache in a JSON file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save semantic cache: {e}")

    def init_model(self):
        """Initializes the SentenceTransformer embedder model."""
        try:
            from sentence_transformers import SentenceTransformer
            # Using a highly-optimized, lightweight model suitable for CPU execution
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print("SentenceTransformer 'all-MiniLM-L6-v2' successfully loaded for semantic caching.")
        except Exception as e:
            print(f"SentenceTransformer not initialized, falling back to keyword similarity: {e}")
            self.model = None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Helper to calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def lookup(self, query: str) -> Optional[str]:
        """
        Looks up a query in the cache. Returns the cached response if similarity exceeds threshold.
        """
        if not self.cache_data:
            return None

        # Clean query
        query_clean = query.strip()

        # Semantic cosine similarity matching
        if self.model:
            try:
                query_emb = self.model.encode(query_clean)
                best_score = -1.0
                best_response = None
                
                for entry in self.cache_data:
                    entry_query = entry.get("query", "")
                    
                    # Compute embedding for entry query if missing or dynamically loaded
                    if "embedding" not in entry or entry["embedding"] is None:
                        entry["embedding"] = self.model.encode(entry_query).tolist()
                        self.save_cache() # Persist the added embedding
                    
                    entry_emb = np.array(entry["embedding"])
                    score = self._cosine_similarity(query_emb, entry_emb)
                    
                    if score > best_score:
                        best_score = score
                        best_response = entry.get("response")
                
                if best_score >= self.threshold:
                    print(f"[Semantic Cache HIT] Cosine Similarity: {best_score:.4f} (Threshold: {self.threshold})")
                    return best_response
                else:
                    print(f"[Semantic Cache MISS] Best Similarity: {best_score:.4f} (Threshold: {self.threshold})")
            except Exception as e:
                print(f"Error performing semantic lookup: {e}. Falling back to string checking.")

        # Robust string matching fallback
        query_lower = query_clean.lower()
        for entry in self.cache_data:
            cached_query = entry.get("query", "").lower().strip()
            if query_lower == cached_query or (len(query_lower) > 8 and cached_query in query_lower):
                print("[Semantic Cache HIT] Exact/Substring Match Fallback")
                return entry.get("response")
                
        return None

    def update(self, query: str, response: str):
        """
        Adds a new query and response pair to the cache, computing its embedding if available.
        """
        query_clean = query.strip()
        response_clean = response.strip()

        # Avoid duplicates
        for entry in self.cache_data:
            if entry.get("query", "").lower().strip() == query_clean.lower():
                return

        new_entry = {
            "query": query_clean,
            "response": response_clean
        }

        if self.model:
            try:
                new_entry["embedding"] = self.model.encode(query_clean).tolist()
            except Exception as e:
                print(f"Failed to generate embedding for cache entry: {e}")
                new_entry["embedding"] = None

        self.cache_data.append(new_entry)
        self.save_cache()
        print(f"[Semantic Cache UPDATED] Cached query: '{query_clean[:50]}...'")
