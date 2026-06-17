Class {
	#name : 'CMGame',
	#superclass : 'CMSimulation',
	#instVars : [
		'server',
		'portNumber'
	],
	#category : 'Cormas-GameServer',
	#package : 'Cormas-GameServer'
}

{ #category : 'accessing' }
CMGame class >> defaultPortNumber [

	^ 8765
]

{ #category : 'initialization' }
CMGame >> initialize [

	super initialize.
	portNumber := self class defaultPortNumber.
	
	server := CMGameServer new.
]

{ #category : 'initialization' }
CMGame >> initializeGame [

	self initializeSimulation.

	server startOn: portNumber.
	
	"server whenDataReceivedDo: [ :dict :clientId |
		cormasModel respondToClient: dict ]."
]

{ #category : 'accessing' }
CMGame >> portNumber: aNumber [

	portNumber := aNumber
]

{ #category : 'as yet unclassified' }
CMGame >> stopServer [

	server stop
]
