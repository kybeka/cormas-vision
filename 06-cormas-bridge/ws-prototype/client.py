import asyncio
import websockets
import json

from mock import simulate_input

async def run_client():
    uri = "ws://localhost:8081/ws"

    # try:

    async with websockets.connect(uri) as websocket:
        print("Connected to server")

        # A required handshake message (optional id param to reconnect)
        await websocket.send(json.dumps({'type': 'bonjour'}))

        while True:
            # data = {"occupiedCells": simulate_input()}
            data = {
                    "blue-pawn":   [8, 10],
                    # "red-pawn":    [9, 10, 11, 12],
                    "yellow-pawn": [8, 10, 12]
                    # "white-pawn":  [17, 18, 19, 20]
                }

            print(data)

            await websocket.send(json.dumps(data))
            await asyncio.sleep(3.0)

    # except websockets.ConnectionClosedOK:
    #     print("Server closed the connection cleanly.")

    # except websockets.ConnectionClosedError as e:
    #     print("Server closed with error:", e)

if __name__ == "__main__":
    asyncio.run(run_client())