import asyncio
import websockets
import cv2
from hand_recognizer_class import Recognizer

cap = cv2.VideoCapture(0)

rec = Recognizer()


async def handler(websocket):
    while True:
        ret, frame = cap.read()
        await asyncio.sleep(0.01)
        res = rec.recognize(frame)
        if res is not None:
            await websocket.send(res)


async def main():
    async with websockets.serve(handler, "", 8001):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
