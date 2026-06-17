Extension { #name : 'CMAbstractModel' }

{ #category : '*Cormas-GameServer' }
CMAbstractModel >> respondToClient: aDictionary [
	"Override this method to specify the response to a player action received over the network. aDictionary represents a JSON input"
	
	aDictionary traceCr
]
