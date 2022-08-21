import asyncio
import websockets


async def consumer(message):
    await asyncio.sleep(1)
    print(f"im consumer: {message}")


async def producer():
    print("im producer")
    await asyncio.sleep(1)
    return "shit"


async def consumer_handler(websocket):
    async for message in websocket:
        await consumer(message)


async def producer_handler(websocket):
    while True:
        message = await producer()
        await websocket.send(message)


async def handler(websocket):
    consumer_task = asyncio.create_task(consumer_handler(websocket))
    producer_task = asyncio.create_task(producer_handler(websocket))
    done, pending = await asyncio.wait(
        [consumer_task, producer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()


async def main():
    async with websockets.serve(handler, "", 8001):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
