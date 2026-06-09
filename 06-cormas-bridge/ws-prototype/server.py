import asyncio
import websockets

client = None # Only one client can be connected


async def handler(websocket):
    global client

    if client is not None:
        # Reject new connections
        await websocket.send('Server is busy. Only one client allowed')
        await websocket.close()
        print('Rejected a connection (server busy)')
        return

    client = websocket
    print('Client connected')

    try:
        async for message in websocket:
            print(f'Received: {message}')

    except websockets.ConnectionClosed:
        print('Client disconnected')

    finally:
        client = None # Allow next client to connect


async def main():
    async with websockets.serve(handler, 'localhost', 8765):
        print('Server running on ws://localhost:8765 (only one client allowed)')
        await asyncio.Future() # Run forever


if __name__ == '__main__':
    asyncio.run(main())