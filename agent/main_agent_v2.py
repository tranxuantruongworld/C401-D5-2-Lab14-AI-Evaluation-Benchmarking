import asyncio
import json
import os
import re
import sys
from textwrap import dedent

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import AsyncOpenAI

from engine.utils import retry_with_exponential_backoff

load_dotenv()


class MainAgentV2:
    """Support Agent sử dụng kiến trúc RAG (Retrieval-Augmented Generation) chuyên nghiệp.
    Sử dụng Two-Step Pipeline:
    1. LLM-based Filtering (Retriever): Chọn lọc top K document IDs liên quan.
    2. Context-Augmented Generation (Generator): Sinh câu trả lời dựa trên context được lọc.
    """

    def __init__(self):
        self.name = 'SupportAgent-v3-Retriever'
        self.client = AsyncOpenAI(
            api_key=os.getenv('GOOGLE_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL'),
        )
        self.context_file = 'data/context.md'
        self.chunks = self._load_context()
        self.embed_model = TextEmbedding()

        print('Indexing knowledge base...')
        self.db_embeddings = list(
            self.embed_model.embed([c['content'] for c in self.chunks])
        )

    def _load_context(self) -> list[dict]:
        """Tải và phân tách context.md thành các section dựa trên số ID."""
        if not os.path.exists(self.context_file):
            print(f'Warning: {self.context_file} not found.')
            return []

        try:
            with open(self.context_file, encoding='utf-8') as f:
                content = f.read()

            pattern = r'## (\d+)\.\s*(.*?)(?=\n## \d+\.|$)'
            matches = re.findall(pattern, content, re.DOTALL)

            chunks = []
            for id_, text in matches:
                chunks.append({'id': id_, 'content': text.strip()})
            return chunks
        except Exception as e:
            print(f'Error loading context: {e}')
            return []

    @retry_with_exponential_backoff(base_delay=10.0, max_retries=5)
    async def retrieve_ids(self, question: str) -> list[str]:
        """Bước 1: Semantic Retrieval + LLM Re-ranking."""
        # 1. Semantic Search
        query_embedding = list(self.embed_model.embed([question]))[0]
        scores = []
        for doc_emb in self.db_embeddings:
            score = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            scores.append(score)

        # Get top 15 candidates
        top_indices = np.argsort(scores)[-7:][::-1]
        candidates = [self.chunks[i] for i in top_indices]

        # 2. LLM Re-ranker
        return [c['id'] for c in candidates]

    @retry_with_exponential_backoff(base_delay=10.0, max_retries=5)
    async def query(self, question: str) -> dict:
        """Xử lý câu hỏi qua Two-Step Pipeline."""
        if not self.chunks:
            return {
                'answer': 'Hệ thống hiện không có dữ liệu ngữ cảnh.',
                'contexts': [],
                'metadata': {'error': 'No context available'},
            }

        # --- STEP 1: RETRIEVAL ---
        retrieved_ids = await self.retrieve_ids(question)

        # Pull actual content for the retrieved IDs
        selected_chunks = [c for c in self.chunks if c['id'] in retrieved_ids]
        context_str = '\n'.join(
            [f'Document {c["id"]}:\n{c["content"]}' for c in selected_chunks]
        )

        # --- STEP 2: GENERATION ---
        prompt = dedent(f"""
        <instruction>
            Use the provided context below to answer the question.
            Be concise and scientifically accurate.
        </instruction>

        <context>
        {context_str}
        </context>

        <question>
        {question}
        </question>

        <output_format>
        Return a JSON with schema:
        {{
            "answer": "Your detailed scientific answer...",
            "reasoning": "Brief explanation of why these documents were chosen."
        }}
        </output_format>
        """)

        response = await self.client.chat.completions.create(
            model='gemini-3.1-flash-lite-preview',
            messages=[{'role': 'user', 'content': prompt}],
            response_format={'type': 'json_object'},
        )

        try:
            result = json.loads(response.choices[0].message.content)

            # Format context with "Trích dẫn" prefix for the evaluator
            final_contexts = [
                f'[Trích dẫn {c["id"]}]: {c["content"]}' for c in selected_chunks
            ]

            return {
                'answer': result.get('answer', 'Không tìm thấy câu trả lời.'),
                'contexts': final_contexts,
                'metadata': {
                    'model': 'gemini-3.1-flash-lite',
                    'chunk_ids': retrieved_ids,
                    'tokens_used': response.usage.total_tokens,
                },
            }
        except Exception as e:
            return {
                'answer': f'Xin lỗi, đã có lỗi xảy ra trong quá trình sinh câu trả lời: {str(e)}',
                'contexts': [],
                'metadata': {'error': str(e), 'chunk_ids': retrieved_ids},
            }


if __name__ == '__main__':

    async def test():
        agent = MainAgentV2()
        res = await agent.query('Hit Rate (Recall@K) là gì?')
        print(json.dumps(res, indent=2, ensure_ascii=False))

    asyncio.run(test())
