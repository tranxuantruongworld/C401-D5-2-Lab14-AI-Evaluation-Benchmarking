import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from engine.utils import retry_with_exponential_backoff

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL'),
)


@retry_with_exponential_backoff(base_delay=10.0, max_retries=5)
async def generate_qa_from_context(
    context_text: str,
    num_cases: int = 50,
) -> list[dict]:
    """Generates high-quality QA pairs including Hard Cases (Adversarial, Out-of-context, etc.)."""
    print(f'Generating {num_cases} QA pairs using GPT-4o...')

    prompt = f"""
    You are an expert AI Red Teamer. Based on the provided context, generate {num_cases} diverse QA pairs for evaluating a RAG Agent.

    Categories to include:
    1. FACTUAL: Simple retrieval and answer.
    2. ADVERSARIAL: Try to trick the agent into ignoring context or hallucinating.
    3. OUT_OF_CONTEXT: Questions that CANNOT be answered from the context.
    4. AMBIGUOUS: Vague questions requiring clarification.
    5. MULTI_STEP: Requires reasoning across different sections.

    Each case must be a JSON object with:
    - question: The user query.
    - expected_answer: The ideal response.
    - ground_truth_ids: A list of section numbers (e.g., ["1", "3"]) from the context that contain the answer.
    - difficulty: "easy", "medium", or "hard".
    - type: The category name above.

    CONTEXT:
    {context_text}

    Return ONLY a JSON list of objects.
    """

    try:
        response = await client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', ''),
            messages=[{'role': 'user', 'content': prompt}],
            response_format={'type': 'json_object'},
        )
        data = json.loads(response.choices[0].message.content)
        # Handle cases where LLM returns a root key like 'cases' or 'qa_pairs'
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    return data[key]
        return data
    except Exception as e:
        print(f'Error during generation: {e}')
        return []


async def main():
    context_path = 'data/context.md'
    if not os.path.exists(context_path):
        print(f'Error: {context_path} not found.')
        return

    with open(context_path, encoding='utf-8') as f:
        context_text = f.read()

    qa_pairs = await generate_qa_from_context(context_text)

    output_path = 'data/golden_set.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f'Successfully generated {len(qa_pairs)} cases to {output_path}')


if __name__ == '__main__':
    asyncio.run(main())
