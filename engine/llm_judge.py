import google.generativeai as genai
import json
import asyncio

genai.configure(api_key="PLEASE ADD GEMMINI_API_KEY")

class MultiModelJudge:
    def __init__(self):
        # Model mạnh đóng vai trò Trưởng ban giám khảo
        self.senior_judge = genai.GenerativeModel('gemma-3-27b-it')
        # Model nhẹ hơn đóng vai trò Giám khảo hỗ trợ
        self.junior_judge = genai.GenerativeModel('gemma-3-4b-it')

    async def _get_individual_score(self, model, role_name, q, a, gt):
        """Hàm gọi từng model để chấm điểm độc lập"""
        prompt = f"""
        Bạn là giám khảo AI {role_name}. Hãy chấm điểm câu trả lời dựa trên Ground Truth (Đáp án đúng).
        
        CÂU HỎI: {q}
        CÂU TRẢ LỜI CỦA AGENT: {a}
        ĐÁP ÁN ĐÚNG (GT): {gt}

        Quy tắc:
        - Chấm trên thang điểm 5.
        - Trả về JSON: {{"score": float, "reasoning": "giải thích ngắn gọn"}}
        """
        try:
            response = await model.generate_content_async(prompt)
            # Làm sạch JSON (giống logic ở các phần trước)
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            return json.loads(text)
        except Exception as e:
            print(f"Lỗi từ {role_name}: {e}")
            return {"score": 0.0, "reasoning": "Lỗi API"}

    async def evaluate_multi_judge(self, q, a, gt):
        """Quy trình đánh giá đa mô hình"""
        # Chạy song song cả 2 model để tiết kiệm thời gian
        results = await asyncio.gather(
            self._get_individual_score(self.senior_judge, "Cao cấp", q, a, gt),
            self._get_individual_score(self.junior_judge, "Cơ bản", q, a, gt)
        )

        score1, score2 = results[0]["score"], results[1]["score"]
        reason1, reason2 = results[0]["reasoning"], results[1]["reasoning"]

        # 1. Tính toán Điểm cuối (Trung bình cộng)
        final_score = (score1 + score2) / 2

        # 2. Tính toán Độ đồng thuận (Agreement Rate)
        # Công thức: 1 - (khoảng cách điểm / thang điểm tối đa)
        agreement_rate = 1 - (abs(score1 - score2) / 5)

        # 3. Tổng hợp reasoning
        combined_reasoning = f"Judge 27B: {reason1} | Judge 4B: {reason2}"

        return {
            "final_score": round(final_score, 2),
            "agreement_rate": round(agreement_rate, 2),
            "individual_scores": {"senior": score1, "junior": score2},
            "reasoning": combined_reasoning
        }

# --- Cách dùng ---
async def main():
    judge = MultiModelJudge()
    # Đúng: await nằm trong hàm async
    result = await judge.evaluate_multi_judge(
        "10 triệu mua máy gì?", 
        "Nên mua Galaxy A55", 
        "Galaxy A55 là lựa chọn tốt nhất tầm 10tr"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())