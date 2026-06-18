"Load the full Planet-C + Cormas-Vision stack into a fresh Pharo 13 image.
 Run in a Playground with 'Do it all' (Ctrl+D after selecting all).

 Dependency chain:
   kybeka/cormas               -> CORMAS framework (user's fork)
   cormas/planetc-model        -> PlanetC-Model  (the game model)
   cormas/cormas-game-server   -> CMGameServer   (WebSocket server base)
   kybeka/cormas-vision-server -> CMVisionServerCommand  (CV bridge UI button)
"

"1. CORMAS framework (user's fork)"
Metacello new
  repository: 'github://kybeka/cormas:dev';
  baseline: 'Cormas';
  load.

"2. Planet-C model"
Metacello new
  baseline: 'PlanetC';
  repository: 'github://cormas/planetc-model:main';
  load.

"3. CV vision server (bridge + UI toggle button for the CORMAS space presenter)"
Metacello new
  baseline: 'CormasVisionServer';
  repository: 'github://kybeka/cormas-vision-server:main';
  onConflict: [ :e | e useIncoming ];
  onUpgrade: [ :e | e useIncoming ];
  onWarning: [ :e | e load ];
  ignoreImage;
  load.

"4. After loading: open the Planet-C model, hit Initialize simulation,
   then use the Start Server / Stop Server toggle button in the CORMAS toolbar.
   The server listens on ws://localhost:8081/ws.

   Python side:
     python demo.py --ws-url ws://localhost:8081/ws
"
