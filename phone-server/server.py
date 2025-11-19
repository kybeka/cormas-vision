import asyncio
import websockets
import base64
import re
import os

os.makedirs("frames", exist_ok=True)
data_url_pattern = re.compile(r"^data:image/\w+;base64,")
frame_counter = 0

async def handler(websocket):
    global frame_counter
    print("Client connected")

    try:
        async for message in websocket:
            if message.startswith("data:image"):
                b64_data = data_url_pattern.sub("", message)
                img_bytes = base64.b64decode(b64_data)

                filename = f"frames/frame_{frame_counter:05d}.jpg"
                with open(filename, "wb") as f:
                    f.write(img_bytes)

                frame_counter += 1
            else:
                print("Non-image message:", message)
    except websockets.exceptions.ConnectionClosed as e:
        print("Client disconnected:", e)

async def main():
    async with websockets.serve(
        handler,
        "0.0.0.0",
        8080,              # <-- plain WS on port 8080
        ping_interval=None
    ):
        print("WebSocket on ws://0.0.0.0:8080/ws")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())


# import asyncio
# import websockets
# import base64
# import re
# import os
# import ssl

# # ---------- SSL setup ----------
# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

# # ---------- Frame handling setup ----------
# os.makedirs("frames", exist_ok=True)
# data_url_pattern = re.compile(r"^data:image/\w+;base64,")
# frame_counter = 0

# async def handler(websocket):
#     global frame_counter
#     print("Client connected")

#     try:
#         async for message in websocket:
#             if message.startswith("data:image"):
#                 # Strip "data:image/jpeg;base64," prefix
#                 b64_data = data_url_pattern.sub("", message)
#                 img_bytes = base64.b64decode(b64_data)

#                 # Save frame (or swap with your processing)
#                 filename = f"frames/frame_{frame_counter:05d}.jpg"
#                 with open(filename, "wb") as f:
#                     f.write(img_bytes)

#                 frame_counter += 1
#                 # Optional: confirm
#                 # await websocket.send(f"OK {frame_counter}")
#             else:
#                 print("Non-image message:", message)
#     except websockets.exceptions.ConnectionClosed as e:
#         print("Client disconnected:", e)

# async def main():
#     # Note: port 8081 and *secure* WebSocket (wss://)
#     async with websockets.serve(
#         handler,
#         "0.0.0.0",
#         8081,
#         ssl=ssl_context,
#         ping_interval=None,
#     ):
#         print("Secure WebSocket on wss://0.0.0.0:8081/ws")
#         await asyncio.Future()  # run forever

# if __name__ == "__main__":
#     asyncio.run(main())