import json
import asyncio
import os
import google.generativeai as genai
from typing import List, Dict

# Cấu hình API Key (Lưu ý: Không nên để lộ Key công khai nhé!)
genai.configure(api_key="PLEASE ADD GEMMINI_API_KEY")

async def generate_qa_from_text(text: str, num_pairs: int = 5) -> List[Dict]:
    model = genai.GenerativeModel('gemma-3-27b-it')
    
    # Prompt chuyên biệt cho Tư vấn điện thoại
    prompt = f"""
    Bạn là một chuyên gia tư vấn smartphone cao cấp. Dựa vào văn bản kiến thức dưới đây, 
    hãy tạo ra {num_pairs} cặp (Câu hỏi của khách hàng, Câu trả lời kỳ vọng).

    Yêu cầu:
    1. Câu hỏi (question): Phải mô phỏng ngôn ngữ tự nhiên của người mua (Ví dụ: "Mình có 10 triệu thì mua máy gì chụp ảnh đẹp?").
    2. Câu trả lời (expected_answer): Phải chuyên nghiệp, có so sánh và đưa ra lựa chọn cụ thể dựa trên kiến thức cung cấp.
    3. Metadata: Phân loại độ khó (easy, medium, hard) và phân khúc (giá rẻ, cận cao cấp, flagship).
    4. Định dạng trả về: Chỉ trả về JSON list, không có text thừa.

    Kiến thức để tư vấn:
    {text}
    """

    try:
        response = await model.generate_content_async(prompt)
        clean_text = response.text.strip()
        
        # Xử lý bóc tách JSON từ Markdown nếu có
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()
            
        return json.loads(clean_text)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return []

async def main():
    # Nội dung kiến thức làm nền tảng để Model sinh câu hỏi/trả lời
    raw_knowledge = """
    1. Phân khúc Giá rẻ (Dưới 5 triệu): Samsung Galaxy A15 (màn hình AMOLED đẹp), Redmi Note 13 (sạc nhanh, cấu hình tốt trong tầm giá).
    2. Phân khúc Tầm trung (5 - 10 triệu): iPhone 11 (hiệu năng ổn định nhưng màn hình cũ), Samsung Galaxy A55 (thiết kế cao cấp, kháng nước), Xiaomi 13T (Camera Leica, cấu hình mạnh).
    3. Phân khúc Cận cao cấp (10 - 15 triệu): iPhone 13 (giữ giá tốt, quay phim đẹp), Galaxy S23 FE (tính năng flagship rút gọn).
    4. Phân khúc Flagship (Trên 20 triệu): iPhone 15 Pro Max (chip A17 Pro, vỏ titan), Samsung Galaxy S24 Ultra (Bút S-Pen, AI dịch thuật, zoom 100x).
    Lưu ý: Nếu khách cần chụp ảnh, ưu tiên Samsung dòng S hoặc iPhone dòng Pro. Nếu khách chơi game, ưu tiên các máy chạy chip Snapdragon dòng 8 hoặc Apple Silicon.
    """
    
    # Ở đây mình để 10 cặp để test, bạn có thể tăng lên tùy nhu cầu
    qa_pairs = await generate_qa_from_text(raw_knowledge, 10)
    
    os.makedirs("data", exist_ok=True)
    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
    print(f"✅ Đã tạo xong {len(qa_pairs)} mẫu tư vấn! Lưu tại: data/golden_set.jsonl")

if __name__ == "__main__":
    asyncio.run(main())