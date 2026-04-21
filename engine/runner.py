import asyncio
import json
import time

from tqdm.asyncio import tqdm

from agent.main_agent import MainAgent
from agent.main_agent_v2 import MainAgentV2
# from engine.expert_eval import ExpertEvaluator
from engine.llm_judge import MultiModelJudge
from engine.retrieval_eval import RetrievalEvaluator


class BenchmarkRunner:
    def __init__(self, agent: MainAgent | MainAgentV2, max_concurrent: int = 3):
        self.agent = agent
        self.retrieval_eval = RetrievalEvaluator()
        self.judge = MultiModelJudge()
        # self.expert_judge = ExpertEvaluator()
        # With 15 RPM, we should be very conservative.
        # Each case does 1 agent call + 2 judge calls = 3 calls/case.
        # 3 concurrent cases = 9 calls in a burst.
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_single_case(self, case: dict) -> dict:
        async with self.semaphore:
            start_time = time.perf_counter()

            # 1. Get Agent Response
            agent_resp = await self.agent.query(case['question'])

            # 2. Retrieval Metrics
            retrieved_ids = self.retrieval_eval.extract_ids(agent_resp)
            mrr = self.retrieval_eval.calculate_mrr(
                retrieved_ids, case['ground_truth_ids'],
            )
            hit_rate = self.retrieval_eval.calculate_hit_rate(
                retrieved_ids,
                case['ground_truth_ids'],
            )

            # 3. Generation Metrics (Multi-Judge)
            judge_result = await self.judge.evaluate_multi_judge(
                question=case['question'],
                answer=agent_resp['answer'],
                expected=case['expected_answer'],
                # context='\n'.join(agent_resp['contexts']),
            )

            latency = time.perf_counter() - start_time

            return {
                'test_case': case['question'],
                'agent_response': agent_resp['answer'],
                'type': case['type'],
                'latency': latency,
                'mrr': mrr,
                'hit_rate': hit_rate,
                **judge_result,
            }

    async def run_benchmark(self, golden_set: list[dict]) -> list[dict]:
        tasks = [self.evaluate_single_case(case) for case in golden_set]
        return await tqdm.gather(*tasks)

        # summary = {
        #     'metadata': {
        #         'total': len(results),
        #         'version': self.agent.name,
        #         'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        #     },
        #     'metrics': {
        #         'avg_mrr': sum(r['mrr'] for r in results) / len(results),
        #         'hit_rate': sum(r['hit_rate'] for r in results) / len(results),
        #         'avg_score': sum(r['final_score'] for r in results) / len(results),
        #         'agreement_rate': sum(r['is_agreement'] for r in results)
        #         / len(results),
        #         'avg_latency': sum(r['latency'] for r in results) / len(results),
        #     },
        # }
        #
        # return results, summary
