import numpy as np
import google.generativeai as genai
import json
genai.configure(api_key="PLEASE ADD GEMMINI_API_KEY")

class ExpertEvaluator:
    def __init__(self):
        # Sử dụng Gemma-3 để làm "Judge" chấm điểm Faithfulness và Relevancy
        self.judge_model = genai.GenerativeModel('gemma-3-27b-it')

    async def score(self, case: dict, resp: dict) -> dict:
        """
        case: Chứa 'expected_answer' và 'context' (Golden Set)
        resp: Chứa 'answer' và 'contexts' từ Agent
        """
        # 1. Tính toán Retrieval Metrics (Toán học)
        retrieval_results = self._calculate_retrieval_metrics(
            expected_context=case.get("context", ""),
            retrieved_contexts=resp.get("contexts", [])
        )

        # 2. Tính toán Generation Metrics (LLM-as-a-judge)
        generation_results = await self._calculate_generation_metrics(
            question=case.get("question", ""),
            context="\n".join(resp.get("contexts", [])),
            answer=resp.get("answer", "")
        )

        return {
            **generation_results,
            "retrieval": retrieval_results
        }

    def _calculate_retrieval_metrics(self, expected_context: str, retrieved_contexts: list) -> dict:
        """
        Tính Hit Rate và MRR dựa trên việc kiểm tra xem đoạn văn bản kỳ vọng 
        có nằm trong danh sách được lấy ra không.
        """
        hit = 0
        mrr = 0
        
        for i, ctx in enumerate(retrieved_contexts):
            # Kiểm tra xem context kỳ vọng có xuất hiện trong kết quả (matching cơ bản)
            if expected_context[:50] in ctx: # So khớp 50 ký tự đầu để định danh
                hit = 1
                mrr = 1 / (i + 1)
                break
        
        return {"hit_rate": float(hit), "mrr": float(mrr)}

    async def _calculate_generation_metrics(self, question, context, answer) -> dict:
        """
        Sử dụng Gemma-3 để chấm điểm độ trung thực và độ liên quan.
        """
        prompt = f"""
        Bạn là một chuyên gia kiểm định AI. Hãy đánh giá câu trả lời dựa trên ngữ cảnh.
        
        CÂU HỎI: {question}
        NGỮ CẢNH: {context}
        CÂU TRẢ LỜI: {answer}

        Hãy trả về JSON với 2 chỉ số từ 0.0 đến 1.0:
        1. faithfulness: Câu trả lời có đúng với thông tin trong ngữ cảnh không? (0: sai hoàn toàn, 1: chính xác tuyệt đối)
        2. relevancy: Câu trả lời có giải quyết đúng trọng tâm câu hỏi không?

        JSON:
        """
        
        try:
            response = await self.judge_model.generate_content_async(prompt)
            # Làm sạch text để parse JSON
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            
            res = json.loads(text)
            return {
                "faithfulness": res.get("faithfulness", 0.0),
                "relevancy": res.get("relevancy", 0.0)
            }
        except:
            return {"faithfulness": 0.0, "relevancy": 0.0}