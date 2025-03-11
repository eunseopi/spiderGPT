import asyncio
import websockets

async def hello():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # 서버로 메시지 전송
        await websocket.send("안녕하세요, 서버!")

        # 서버로부터 응답 받기
        response = await websocket.recv()
        print(f"서버 응답: {response}")

# 클라이언트 실행
if __name__ == "__main__":
    asyncio.run(hello())
