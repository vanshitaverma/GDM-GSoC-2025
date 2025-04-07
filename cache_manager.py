# code scaffolding for the cache manager

import hashlib
import json
import os

class CacheManager:
    def __init__(self, cache_dir="cache/"):
        os.makedirs(cache_dir, exist_ok=True)
        self.chunk_cache_path = os.path.join(cache_dir, "transcript_chunks.json")
        self.response_cache_path = os.path.join(cache_dir, "qa_response_cache.json")
        self.chunk_cache = self._load_json(self.chunk_cache_path)
        self.response_cache = self._load_json(self.response_cache_path)

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_json(self, data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_chunk(self, chunk_id, content):
        self.chunk_cache[chunk_id] = content
        self._save_json(self.chunk_cache, self.chunk_cache_path)

    def get_chunk(self, chunk_id):
        return self.chunk_cache.get(chunk_id)

    def get_all_chunks(self):
        return self.chunk_cache

    def generate_chunk_id(self, text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def cache_response(self, question, context_ids, answer):
        key = self._generate_response_key(question, context_ids)
        self.response_cache[key] = answer
        self._save_json(self.response_cache, self.response_cache_path)

    def get_cached_response(self, question, context_ids):
        key = self._generate_response_key(question, context_ids)
        return self.response_cache.get(key)

    def _generate_response_key(self, question, context_ids):
        key_data = {
            "q": question,
            "ctx": sorted(context_ids)
        }
        return hashlib.sha256(json.dumps(key_data).encode()).hexdigest()
