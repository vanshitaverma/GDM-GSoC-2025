# code scaffolding for the context builder

from typing import List, Dict
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from .cache_manager import CacheManager

class ContextBuilder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.cache_manager = CacheManager()
        self.chunk_embeddings = self._compute_chunk_embeddings()

    def _compute_chunk_embeddings(self):
        chunks = self.cache_manager.get_all_chunks()
        embeddings = {
            chunk_id: self.embedder.encode(chunk, convert_to_tensor=True)
            for chunk_id, chunk in chunks.items()
        }
        return embeddings

    def find_relevant_chunks(self, question, top_k=3) -> List[str]:
        q_embedding = self.embedder.encode(question, convert_to_tensor=True)
        scores = []
        for chunk_id, emb in self.chunk_embeddings.items():
            sim = util.pytorch_cos_sim(q_embedding, emb).item()
            scores.append((chunk_id, sim))
        top_chunks = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return [chunk_id for chunk_id, _ in top_chunks]

    def build_context(self, question, include_visual=False, visual_context_path=None):
        chunk_ids = self.find_relevant_chunks(question)
        chunks = [self.cache_manager.get_chunk(cid) for cid in chunk_ids]

        context = "\n---\n".join(chunks)

        if include_visual and visual_context_path and os.path.exists(visual_context_path):
            with open(visual_context_path, "r") as f:
                visual_data = json.load(f)
            visual_context = self._get_relevant_visuals(question, visual_data)
            context += "\n\n[Visual Descriptions]\n" + "\n".join(visual_context)

        return context, chunk_ids

    def _get_relevant_visuals(self, question, visual_data, top_k=2) -> List[str]:
        visual_texts = [frame["description"] for frame in visual_data]
        q_embedding = self.embedder.encode(question, convert_to_tensor=True)
        frame_embeddings = self.embedder.encode(visual_texts, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(q_embedding, frame_embeddings)[0]
        top_indices = np.argsort(similarities.cpu())[::-1][:top_k]

        return [visual_texts[i] for i in top_indices]
