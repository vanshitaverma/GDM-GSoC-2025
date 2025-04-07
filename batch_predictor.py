# code scaffolding for the batch_predictor.py

import json
from tqdm import tqdm
from src.cache_manager import CacheManager
from src.context_builder import ContextBuilder
from src.gemini_interface import GeminiClient  # I will be defining this based on the Gemini API logic

class BatchPredictor:
    def __init__(self, visual_context_path=None, use_visual=True):
        self.cache = CacheManager()
        self.context_builder = ContextBuilder()
        self.gemini = GeminiClient()
        self.visual_context_path = visual_context_path
        self.use_visual = use_visual

    def load_questions(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def run_batch(self, question_list):
        results = []

        for question in tqdm(question_list, desc="Processing batch"):
            # Building context here
            context, chunk_ids = self.context_builder.build_context(
                question,
                include_visual=self.use_visual,
                visual_context_path=self.visual_context_path
            )

            # Checking cache here
            cached_answer = self.cache.get_cached_response(question, chunk_ids)
            if cached_answer:
                answer = cached_answer
                source = "cache"
            else:
                # Querying Gemini
                answer = self.gemini.query_with_context(question, context)
                self.cache.cache_response(question, chunk_ids, answer)
                source = "gemini"

            results.append({
                "question": question,
                "answer": answer,
                "source": source,
                "chunk_ids": chunk_ids
            })

        return results

    def save_results(self, results, path="outputs/answers_batch.json"):
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
