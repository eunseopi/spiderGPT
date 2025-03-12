import os
import openai
import json
import asyncio
import tiktoken
import uvicorn
from fastapi import FastAPI, WebSocket
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

PORT = int(os.getenv("PORT"))

# OpenAI API Key를 환경 변수에서 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ 환경 변수 'OPENAI_API_KEY'가 설정되지 않았습니다!")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Chroma DB 설정
CHROMA_DB_PATH = "./chroma_db"
tokenizer = tiktoken.get_encoding("cl100k_base")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
model_name = "jhgan/ko-sbert-nli"
hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=hf)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ 클라이언트 연결됨")

    while True:
        try:
            data = await websocket.receive_text()
            json_data = json.loads(data)
            user_message = json_data.get("message", "")

            if not user_message:
                await websocket.send_text(json.dumps({"status": "error", "message": "메시지가 비어있음"}))
                continue

            docs_with_scores = db.similarity_search_with_relevance_scores(user_message, k=3)
            THRESHOLD = 0.4
            filtered_docs = [
                (doc.page_content, doc.metadata.get("source", "알 수 없음"), score)
                for doc, score in docs_with_scores if score >= THRESHOLD
            ]

            if filtered_docs:
                context = "\n\n".join([f"{i+1}. {content}" for i, (content, _, _) in enumerate(filtered_docs)])
                prompt = (
                    f"아래의 기술 문서를 참고하여 사용자의 질문에 답변하세요.\n\n"
                    f"사용자 질문: {user_message}\n\n"
                    f"기술 문서:\n{context}\n\n"
                    f"위 문서를 참고하여 정확한 답변을 작성해 주세요."
                )
            else:
                prompt = f"사용자의 질문: {user_message}\n\n관련 문서를 찾지 못했습니다. 일반적인 답변을 제공해주세요."

            gpt_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )

            response_data = {"status": "success", "response": gpt_response.choices[0].message.content}
            await websocket.send_text(json.dumps(response_data))

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            await websocket.close()
            break

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render는 `PORT` 환경 변수를 자동 설정
    uvicorn.run(app, host="0.0.0.0", port=port)
