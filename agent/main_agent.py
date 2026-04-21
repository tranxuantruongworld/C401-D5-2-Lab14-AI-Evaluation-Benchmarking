import asyncio
import json
import os
import sys
import numpy as np
from typing import List, Dict
from fastembed import TextEmbedding
from openai import AsyncOpenAI

from engine.utils import retry_with_exponential_backoff

# Cấu hình Gemini API Key


class MainAgent:
    def __init__(self):
        self.name = 'Gemma-FastRAG-v1'

        # 1. Khởi tạo LLM (Gemma 3)
        # Nếu model gemma-3-27b-it chưa khả dụng ở vùng của bạn,
        # bạn có thể đổi sang 'gemini-1.5-flash' để test logic.
        self.model = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL'),
        )

        # 2. Khởi tạo Embedding Model siêu nhẹ (FastEmbed)
        # Model mặc định: BAAI/bge-small-en-v1.5 (~67MB)
        self.embed_model = TextEmbedding()

        # 3. Knowledge Base (Cơ sở tri thức mẫu)
        self.documents = [
            'Phân khúc Giá rẻ (Dưới 5 triệu): Samsung Galaxy A15 sở hữu màn hình AMOLED đẹp, trong khi Redmi Note 13 nổi bật với sạc nhanh và cấu hình tốt trong tầm giá.',
            'Phân khúc Tầm trung (5 - 10 triệu): iPhone 11 có hiệu năng ổn định nhưng màn hình cũ. Samsung Galaxy A55 có thiết kế cao cấp và kháng nước. Xiaomi 13T mạnh về cấu hình và sở hữu Camera Leica.',
            'Phân khúc Cận cao cấp (10 - 15 triệu): iPhone 13 là lựa chọn giữ giá tốt và quay phim đẹp. Galaxy S23 FE cung cấp các tính năng của dòng flagship nhưng được rút gọn chi phí.',
            'Phân khúc Flagship (Trên 20 triệu): iPhone 15 Pro Max sử dụng chip A17 Pro và vỏ titan siêu bền. Samsung Galaxy S24 Ultra hỗ trợ Bút S-Pen, AI dịch thuật và khả năng zoom 100x.',
            'Tư vấn chụp ảnh: Nếu khách hàng ưu tiên chụp ảnh, hệ thống nên đề xuất các dòng Samsung Galaxy S hoặc iPhone dòng Pro để có chất lượng ống kính tốt nhất.',
            'Tư vấn chơi game: Đối với nhu cầu chơi game nặng, hãy ưu tiên các thiết bị chạy chip Snapdragon dòng 8 (như trên các máy Android cao cấp) hoặc chip Apple Silicon (trên iPhone).',
            'Lưu ý về ngân sách: Khi khách hàng có ngân sách cố định, cần đối chiếu với các phân khúc Giá rẻ (<5tr), Tầm trung (5-10tr), Cận cao cấp (10-15tr) và Flagship (>20tr).',
        ]

        # 4. Tiền tính toán Embedding cho database (Index)
        print('Indexing knowledge base...')
        self.db_embeddings = list(self.embed_model.embed(self.documents))

    async def _retrieve(self, query: str, top_k: int = 2) -> List[str]:
        """Tìm kiếm ngữ nghĩa sử dụng Cosine Similarity."""
        # Tạo embedding cho câu hỏi
        query_embedding = list(self.embed_model.embed([f'query: {query}']))[0]

        # Tính toán độ tương đồng với từng đoạn văn bản
        scores = []
        for doc_emb in self.db_embeddings:
            # Cosine similarity formula
            score = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            scores.append(score)

        # Lấy top_k kết quả có điểm cao nhất
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    @retry_with_exponential_backoff(base_delay=10.0, max_retries=5)
    async def query(self, question: str) -> Dict:
        """Quy trình RAG: Retrieve -> Augment -> Generate."""
        # Bước 1: Retrieval
        relevant_docs = await self._retrieve(question)
        context = '\n'.join([f'- {d}' for d in relevant_docs])

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

        try:
            response = await self.model.chat.completions.create(
                model='gemini-3.1-flash-lite-preview',
                messages=[{'role': 'user', 'content': prompt}],
                response_format={'type': 'json_object'},
            )
            return {
                'answer': response.choices[0].message.content.strip(),
                'contexts': relevant_docs,
                'metadata': {
                    'model': 'gemma-3-27b-it',
                    'retriever': 'fastembed-bge-small',
                    'status': 'success',
                },
            }
        except Exception as e:
            print(f'An error occured: {e}', file=sys.stderr)
            return {'answer': str(e), 'status': 'failed'}


# --- Chạy thử nghiệm ---
# async def main():
#     agent = MainAgent()

#     print("\n--- HỆ THỐNG RAG ĐÃ SẴN SÀNG ---")

#     # Test 1: Câu hỏi có trong dữ liệu
#     q1 = "Làm sao để đánh giá AI?"
#     print(f"\nUser: {q1}")
#     res1 = await agent.query(q1)
#     print(f"Agent: {res1.get('answer')}")

#     # Test 2: Câu hỏi về quy trình đổi mật khẩu
#     # q2 = "Các bước đổi mật khẩu là gì?"
#     # print(f"\nUser: {q2}")
#     # res2 = await agent.query(q2)
#     # print(f"Agent: {res2.get('answer')}")

# if __name__ == "__main__":
#     asyncio.run(main())
