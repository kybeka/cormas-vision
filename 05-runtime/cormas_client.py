import json
import threading
from typing import Dict, List, Optional


class CormasClient:
    """Persistent WebSocket client with one-time handshake and queued sends.

    Usage:
      client = CormasClient("ws://localhost:8081/ws")
      client.start()
      client.send_class_map({"green-token": [6, 11]})
      ...
      client.close()
    """

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self._thread: Optional[threading.Thread] = None
        self._loop = None
        self._queue = None
        self._stop = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="CormasWS", daemon=True)
        self._thread.start()

    def _run_loop(self):
        try:
            import asyncio
        except Exception:
            print("[cormas] asyncio not available; skipping start.")
            return
        import asyncio
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        self._loop.run_until_complete(self._sender_task())

    async def _sender_task(self):
        try:
            import websockets  # type: ignore
        except Exception:
            print("[cormas] websockets not available; cannot start persistent sender.")
            return

        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.ws_url, max_size=1_000_000) as ws:
                    try:
                        await ws.send(json.dumps({"type": "bonjour", "id": "cv-demo"}))
                        print("[cormas] Connected and handshook.")
                    except Exception as e:
                        print(f"[cormas] Handshake failed: {e}")
                        await asyncio.sleep(backoff)
                        continue

                    backoff = 1.0  # reset backoff after successful connect
                    while not self._stop.is_set():
                        class_cells = await self._queue.get()
                        if class_cells is None:  # sentinel
                            return
                        try:
                            await ws.send(json.dumps(class_cells))
                        except Exception as e:
                            print(f"[cormas] Send failed (will reconnect): {e}")
                            break  # drop to outer loop to reconnect
            except Exception as e:
                print(f"[cormas] Connect failed: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    # classes that represent harvesters (pawns) on the board
    PAWN_CLASSES = {"blue-pawn", "red-pawn", "white-pawn", "yellow-pawn",
                    "black-pawn", "green-pawn", "orange-pawn", "pink-pawn"}
    # green tokens = grass/biomass on a cell
    TOKEN_BIOMASS = {"green-token"}
    # yellow tokens = birds (numberOfNewborns)
    TOKEN_BIRDS = {"yellow-token"}
    # orange/red tokens = non-harvest zone / park limit → protected cell
    TOKEN_PROTECTED = {"orange-token", "red-token"}

    def send_frame(self, class_cells: Dict[str, List[int]]):
        """Map CV detections to a CORMAS board-state payload.

        Sends:
          occupiedCells — cells containing any pawn (harvester positions)
          biomass       — cells with green tokens (grass present → biomass: 3)
          birds         — cells with yellow tokens (birds present → numberOfNewborns: 2)
          protected     — cells with orange/red tokens (non-harvest zone)
        Pharo resets all cells each frame then applies these lists.
        """
        occupied, biomass, birds, protected = [], [], [], []
        for cls, cells in class_cells.items():
            if cls in self.PAWN_CLASSES:
                occupied.extend(cells)
            if cls in self.TOKEN_BIOMASS:
                biomass.extend(cells)
            if cls in self.TOKEN_BIRDS:
                birds.extend(cells)
            if cls in self.TOKEN_PROTECTED:
                protected.extend(cells)
        self.send_class_map({
            "occupiedCells": list(dict.fromkeys(occupied)),
            "biomass":       list(dict.fromkeys(biomass)),
            "birds":         list(dict.fromkeys(birds)),
            "protected":     list(dict.fromkeys(protected)),
        })

    def send_class_map(self, class_cells: Dict[str, List[int]]):
        # Enqueue message for background sender; if not running, try to start.
        if not self._thread or not self._thread.is_alive():
            self.start()
        try:
            import asyncio
            if self._loop and self._queue:
                asyncio.run_coroutine_threadsafe(self._queue.put(class_cells), self._loop)
        except Exception as e:
            print(f"[cormas] enqueue failed: {e}")

    def close(self):
        self._stop.set()
        try:
            import asyncio
            if self._loop and self._queue:
                asyncio.run_coroutine_threadsafe(self._queue.put(None), self._loop)  # sentinel
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)