import asyncio
import json
import os
import random
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI

from engine.utils import retry_with_exponential_backoff

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL'),
)


@retry_with_exponential_backoff(base_delay=10.0, max_retries=5)
async def generate_answer(semaphore, question, context_text):
    """Generates an ideal answer for a question based on provided context with retry logic."""
    async with semaphore:
        prompt = f"""
        You are an expert AI evaluator. Based on the provided context, generate a concise and accurate ideal answer for the question.
        
        QUESTION: {question}
        CONTEXT: {context_text}
        
        Return only the answer text.
        """
        response = await client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            messages=[{'role': 'user', 'content': prompt}],
        )
        return response.choices[0].message.content.strip()


async def ingest_beir_scifact():
    print('🚀 Ingesting BEIR/SciFact dataset...')

    # 1. Load Qrels
    print('Loading qrels...')
    qrels_ds = load_dataset('mteb/scifact', split='test')

    # 2. Load Corpus and Queries
    print('Loading corpus and queries...')
    corpus_ds = load_dataset('Beir/scifact', 'corpus', split='corpus')
    queries_ds = load_dataset('Beir/scifact', 'queries', split='queries')

    # Convert to dict for faster access
    corpus_dict = {doc['_id']: doc for doc in corpus_ds}
    queries_dict = {q['_id']: q for q in queries_ds}

    # 3. Select a sample of queries
    # We want 50 cases as per requirements
    target_num_cases = 50
    selected_query_ids = list(set(qrels_ds['query-id']))
    random.seed(42)
    random.shuffle(selected_query_ids)
    selected_query_ids = selected_query_ids[:target_num_cases]

    # 4. Gather relevant qrels and docs
    relevant_qrels = [
        qrel for qrel in qrels_ds if qrel['query-id'] in selected_query_ids
    ]

    # Gather all corpus IDs needed
    needed_corpus_ids = set()
    for qrel in relevant_qrels:
        needed_corpus_ids.add(qrel['corpus-id'])

    # Add some random "noise" documents to the context (up to 100 docs total)
    all_corpus_ids = list(corpus_dict.keys())
    while len(needed_corpus_ids) < 100:
        needed_corpus_ids.add(random.choice(all_corpus_ids))

    needed_corpus_ids = sorted(list(needed_corpus_ids))

    # 5. Prepare context.md
    print(f'Preparing context.md with {len(needed_corpus_ids)} documents...')
    context_content = '# BEIR SciFact Context\n\n'
    id_mapping = {}  # Map original BEIR ID to sequence number for the agent
    for i, cid in enumerate(needed_corpus_ids, 1):
        doc = corpus_dict[cid]
        id_mapping[cid] = str(i)
        title = doc.get('title', '').strip()
        text = doc.get('text', '').strip()
        context_content += f'## {i}. {title}\n{text}\n\n'

    with open('data/context.md', 'w', encoding='utf-8') as f:
        f.write(context_content)

    # 6. Prepare golden_set.jsonl
    print(f'Generating golden_set.jsonl for {len(selected_query_ids)} queries...')
    golden_set = []

    # Limit concurrency to respect rate limits (e.g., 5 parallel requests)
    semaphore = asyncio.Semaphore(5)

    # Pre-calculate mapping: query_id -> list of mapped corpus_ids
    q_to_gt = {}
    for qrel in relevant_qrels:
        qid = qrel['query-id']
        if qid not in q_to_gt:
            q_to_gt[qid] = []
        q_to_gt[qid].append(id_mapping[qrel['corpus-id']])

    tasks = []
    for qid in selected_query_ids:
        question = queries_dict[qid]['text']
        gt_ids = q_to_gt.get(qid, [])
        # Context for answer generation
        gt_context = '\n'.join(
            [
                corpus_dict[cid]['text']
                for cid in needed_corpus_ids
                if id_mapping[cid] in gt_ids
            ]
        )
        tasks.append(generate_answer(semaphore, question, gt_context))

    answers = await asyncio.gather(*tasks)

    for qid, answer in zip(selected_query_ids, answers):
        golden_set.append(
            {
                'question': queries_dict[qid]['text'],
                'expected_answer': answer,
                'ground_truth_ids': q_to_gt.get(qid, []),
                'difficulty': 'medium',
                'type': 'BEIR_SCIFRACT',
            }
        )

    with open('data/golden_set.jsonl', 'w', encoding='utf-8') as f:
        for item in golden_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'✅ Successfully ingested {len(golden_set)} cases from BEIR/SciFact.')
    print('Context saved to data/context.md')
    print('Golden set saved to data/golden_set.jsonl')


if __name__ == '__main__':
    asyncio.run(ingest_beir_scifact())
