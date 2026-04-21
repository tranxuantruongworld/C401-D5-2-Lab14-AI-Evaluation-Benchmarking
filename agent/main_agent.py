import asyncio
import json
import os
import re
from typing import Dict, List

import numpy as np
from fastembed import TextEmbedding
from openai import AsyncOpenAI

from engine.utils import retry_with_exponential_backoff


class MainAgent:
    def __init__(self):
        self.name = 'Gemma-FastRAG-v1'
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

    def _load_context(self) -> List[Dict]:
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

    async def _retrieve(self, query: str, top_k: int = 2) -> List[Dict]:
        """Tìm kiếm ngữ nghĩa sử dụng Cosine Similarity."""
        query_embedding = list(self.embed_model.embed([query]))[0]

        scores = []
        for doc_emb in self.db_embeddings:
            score = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            scores.append(score)

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]

    @retry_with_exponential_backoff(base_delay=10.0, max_retries=5)
    async def query(self, question: str) -> Dict:
        """Quy trình RAG: Retrieve -> Augment -> Generate."""
        # Bước 1: Retrieval
        relevant_chunks = await self._retrieve(question)
        context = '\n'.join([f'- {c["content"]}' for c in relevant_chunks])
        retrieved_ids = [c['id'] for c in relevant_chunks]

        # Bước 2: Prompt Engineering
        prompt = f"""
        Bạn là trợ lý AI chuyên nghiệp. Hãy trả lời câu hỏi dựa trên thông tin dưới đây.
        Nếu thông tin không có trong ngữ cảnh, hãy trả lời 'Tôi không tìm thấy thông tin này'.

        NGỮ CẢNH:
        {context}

        CÂU HỎI:
        {question}

        CÂU TRẢ LỜI:
        """

        response = await self.client.chat.completions.create(
            model='gemini-3.1-flash-lite-preview',
            messages=[{'role': 'user', 'content': prompt}],
            response_format={'type': 'json_object'},
        )

        result = json.loads(response.choices[0].message.content)

        return {
            'answer': result.get('answer', 'Không tìm thấy câu trả lời.'),
            'contexts': [c['content'] for c in relevant_chunks],
            'retrieved_ids': retrieved_ids,
            'metadata': {
                'model': 'gemini-3.1-flash-lite',
                'retriever': 'fastembed-bge-small',
                'status': 'success',
            },
        }
