import asyncio
import json
import os
import re
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI

from engine.utils import retry_with_exponential_backoff

load_dotenv()


class MultiModelJudge:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL'),
        )
        # In a real scenario, Judge B might use a different client (e.g., Anthropic)
        # For this lab, we will use different model versions to simulate Multi-Judge.
        self.judge_a_model = 'gemini-3.1-flash-lite-preview'
        self.judge_b_model = 'gemma-3-27b-it'

    @retry_with_exponential_backoff(base_delay=10.0, max_retries=5)
    async def get_score(
        self,
        model: str,
        question: str,
        answer: str,
        expected: str,
        # context: str,
    ) -> tuple[int, str]:
        prompt = f"""
        <instruction>
        Judge the following ai response on a concrete scale of 1 to 5 for its factual consistency.
        </instruction>

        <question>
        {question}
        </question>

        <ai_answer>
        {answer}
        </ai_answer>

        <expected_answer>
        {expected}
        </expected_answer>

        <return_format>
        Provide only a raw JSON object. Do not include markdown formatting or any other text.
        Schema:
        ```json
        {{
            "score": int,
            "reason" "short explanation"
        }}
        ```
        </return_format>
        """
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
            )
            content = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = re.sub(
                    r'^```(?:json)?\n?|\n?```$',
                    '',
                    content,
                    flags=re.MULTILINE,
                ).strip()

            result = json.loads(content)
            return result['score'], result['reason']
        except Exception as e:
            print(f'An error occurred with model {model}: {e}', file=sys.stderr)
            return 0, 'An error occured'

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        expected: str,
        # context: str,
    ) -> dict:
        # Run both judges in parallel
        results = await asyncio.gather(
            self.get_score(self.judge_a_model, question, answer, expected),
            self.get_score(self.judge_b_model, question, answer, expected),
        )

        score1, score2 = results[0][0], results[1][0]
        reason1, reason2 = results[0][1], results[1][1]

        final_score = (score1 + score2) / 2

        # 2. Tính toán Độ đồng thuận (Agreement Rate)
        # Công thức: 1 - (khoảng cách điểm / thang điểm tối đa)
        agreement_rate = 1 - (abs(score1 - score2) / 5)

        # 3. Tổng hợp reasoning
        combined_reasoning = f'Judge 27B: {reason1} | Judge 4B: {reason2}'

        return {
            'final_score': round(final_score, 2),
            'agreement_rate': round(agreement_rate, 2),
            'individual_scores': {'senior': score1, 'junior': score2},
            'reasoning': combined_reasoning,
        }
