import asyncio
import websockets


async def ask():
    async with websockets.connect('ws://localhost:8001', ping_interval=None) as websocket:
        while True:
            # await asyncio.sleep(0.5)
            # await websocket.send("Timer")
            message = await websocket.recv()
            # await websocket.send("govno")
            print(f"> {message}")


asyncio.ensure_future(ask())

loop = asyncio.get_event_loop().run_forever()
