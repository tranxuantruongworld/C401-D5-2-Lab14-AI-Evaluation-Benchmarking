import asyncio
import json
import os
import time

from agent.main_agent import MainAgent
from agent.main_agent_v2 import MainAgentV2
from engine.runner import BenchmarkRunner

# Giả lập các components Expert
# class ExpertEvaluator:
#     async def score(self, case, resp):
#         # Giả lập tính toán Hit Rate và MRR
#         return {
#             "faithfulness": 0.9,
#             "relevancy": 0.8,
#             "retrieval": {"hit_rate": 1.0, "mrr": 0.5}
#         }

# class MultiModelJudge:
#     async def evaluate_multi_judge(self, q, a, gt):
#         return {
#             "final_score": 4.5,
#             "agreement_rate": 0.8,
#             "reasoning": "Cả 2 model đồng ý đây là câu trả lời tốt."
#         }


async def run_benchmark_with_results(agent_version: int):
    print(f'🚀 Khởi động Benchmark cho Agent V{agent_version}...')

    if not os.path.exists('data/golden_set.jsonl'):
        print(
            "❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước."
        )
        return None, None

    with open('data/golden_set.jsonl', 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print('❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.')
        return None, None

    runner = BenchmarkRunner(
        MainAgent() if agent_version == 1 else MainAgentV2()
    )
    results = await runner.run_benchmark(dataset)

    total = len(results)
    summary = {
        'metadata': {
            'version': agent_version,
            'total': total,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'metrics': {
            'avg_mrr': sum(r['mrr'] for r in results) / len(results),
            'avg_score': sum(r['final_score'] for r in results) / total,
            'hit_rate': sum(r['hit_rate'] for r in results)
            / total,
            'agreement_rate': sum(r['agreement_rate'] for r in results)
            / total,
            'avg_latency': sum(r['latency'] for r in results) / len(results),
            'avg_faithfulness': sum(r['faithfulness'] for r in results) / len(results),
            'avg_relevance': sum(r['relevance'] for r in results) / len(results),
        },
    }
    return results, summary


async def main():
    v1_results, v1_summary = await run_benchmark_with_results(1)

    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results(2)

    if not v1_summary or not v2_summary:
        print('❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.')
        return

    print('\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---')
    metrics_v1 = v1_summary['metrics']
    metrics_v2 = v2_summary['metrics']

    header = f"{'Metric':<20} | {'V1':<10} | {'V2':<10} | {'Delta':<10}"
    separator = "-" * len(header)
    print(header)
    print(separator)

    keys_mapping = [
        ('avg_score', 'Avg Score'),
        ('hit_rate', 'Hit Rate'),
        ('avg_mrr', 'Avg MRR'),
        ('avg_faithfulness', 'Faithfulness'),
        ('avg_relevance', 'Relevance'),
        ('agreement_rate', 'Agreement'),
        ('avg_latency', 'Latency (s)'),
    ]

    for key, label in keys_mapping:
        v1_val = metrics_v1[key]
        v2_val = metrics_v2[key]
        delta = v2_val - v1_val
        print(f"{label:<20} | {v1_val:<10.3f} | {v2_val:<10.3f} | {delta:<+10.3f}")

    print(separator)
    final_delta = metrics_v2['avg_score'] - metrics_v1['avg_score']

    os.makedirs('reports', exist_ok=True)
    with open('reports/summary.json', 'w', encoding='utf-8') as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open('reports/benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if final_delta > 0:
        print('✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)')
    else:
        print('❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)')


if __name__ == '__main__':
    asyncio.run(main())
