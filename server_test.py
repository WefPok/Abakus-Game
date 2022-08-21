import asyncio
import websockets
import cv2
from hand_recognizer_class import recognize

cap = cv2.VideoCapture(0)


async def handler(websocket):
    while True:

        await asyncio.sleep(0.05)

        res = recognize(cap)

        await websocket.send(res)
        if cv2.waitKey(1) == ord('q'):
            break


async def main():
    async with websockets.serve(handler, "", 8001):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
