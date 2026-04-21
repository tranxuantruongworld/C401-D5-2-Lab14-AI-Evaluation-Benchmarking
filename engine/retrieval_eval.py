import re

import numpy as np


class RetrievalEvaluator:
    @staticmethod
    def extract_ids(agent_resp: dict) -> list[str]:
        """Extracts chunk IDs from agent response metadata or context strings."""
        # 1. Try to get from metadata
        ids = agent_resp.get('metadata', {}).get('chunk_ids', [])
        if ids:
            return ids

        # 2. Try to extract from context strings (e.g., "trích dẫn 1")
        extracted_ids = []
        for ctx in agent_resp.get('contexts', []):
            match = re.search(r'trích dẫn (\d+)', ctx)
            if match:
                extracted_ids.append(match.group(1))

        return list(
            dict.fromkeys(extracted_ids)
        )  # Remove duplicates while preserving order

    @staticmethod
    def calculate_hit_rate(
        retrieved_ids: list[str],
        ground_truth_ids: list[str],
        k: int = 3,
    ) -> float:
        """Calculates if at least one ground truth ID is in the top K retrieved results."""
        if not ground_truth_ids:
            return 0.0  # Or 1.0 if we expect nothing to be retrieved

        top_k = retrieved_ids[:k]
        for gt_id in ground_truth_ids:
            if gt_id in top_k:
                return 1.0
        return 0.0

    @staticmethod
    def calculate_mrr(retrieved_ids: list[str], ground_truth_ids: list[str]) -> float:
        """Calculates Mean Reciprocal Rank: 1 / rank of the first relevant document."""
        if not ground_truth_ids:
            return 0.0

        for i, retrieved_id in enumerate(retrieved_ids):
            if retrieved_id in ground_truth_ids:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_batch(
        self,
        batch_retrieved: list[list[str]],
        batch_gt: list[list[str]],
    ) -> dict:
        hit_rates = [
            self.calculate_hit_rate(r, g) for r, g in zip(batch_retrieved, batch_gt, strict=True)
        ]
        mrrs = [self.calculate_mrr(r, g) for r, g in zip(batch_retrieved, batch_gt, strict=True)]

        return {'avg_hit_rate': np.mean(hit_rates), 'avg_mrr': np.mean(mrrs)}
