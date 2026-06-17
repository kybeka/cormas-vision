Extension { #name : 'CMIcons' }

{ #category : '*Cormas-GameServer' }
CMIcons class >> playIcon [
	<script: 'self playIcon inspect'>

	| form |
	self icons at: #playIcon ifPresent: [ :icon | ^ icon ].
	
	form := Form extent: 16 @ 16 depth: 32.
	form getCanvas
		fillColor: Color transparent;
		drawPolygon: { 
				(4 @ 2).
				(12 @ 8).
				(4 @ 14) }
		color: Color magenta
		borderWidth: 0
		borderColor: Color darkGray.
	
	^ self icons at: #playIcon put: form
]
