import asyncio
import websockets
import cv2
from hand_recognizer_class import Analyzer

cap = cv2.VideoCapture(0)

rec = Analyzer()


async def handler(websocket):
    while True:
        ret, frame = cap.read()
        await asyncio.sleep(0.01)
        res = rec.main(frame)
        if res is not None:
            await websocket.send(res)
        if cv2.waitKey(1) == 1:
            break

async def main():
    async with websockets.serve(handler, "", 8001):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
