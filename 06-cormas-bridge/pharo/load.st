"Load the full Planet-C + Cormas-Vision stack into a fresh Pharo 13 image.
 Run this in a Playground with 'Do it all' (Ctrl+D after selecting all).

 Dependency chain:
   cormas/planetc-model        -> PlanetC-Model (the game model)
   cormas/cormas-game-server   -> CMGameServer (WebSocket server base)
   kybeka/cormas               -> CORMAS framework (user's fork)
   this repo / CormasVision    -> CV->Cormas bridge (our package, loaded last)
"

"1. Load upstream dependencies via Metacello"
Metacello new
  baseline: 'PlanetC';
  repository: 'github://cormas/planetc-model:main';
  load.

Metacello new
  baseline: 'CormasGameServer';
  repository: 'github://cormas/cormas-game-server:main';
  load.

"2. Load CormasVision from this repo (Tonel, local path or GitHub)"
"  Option A — from GitHub (once pushed):"
"  Metacello new
     baseline: 'CormasVision';
     repository: 'github://kybeka/cormas-vision:main/06-cormas-bridge/pharo';
     load."

"  Option B — from local clone (during development):"
"  Metacello new
     baseline: 'CormasVision';
     repository: 'tonel:///path/to/cormas-vision/06-cormas-bridge/pharo';
     load."
