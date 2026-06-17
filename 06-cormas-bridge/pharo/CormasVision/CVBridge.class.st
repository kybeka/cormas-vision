Class {
	#name : 'CVBridge',
	#superclass : 'Object',
	#instVars : [
		'server',
		'model',
		'port'
	],
	#category : 'CormasVision',
	#package : 'CormasVision'
}

{ #category : 'instance creation' }
CVBridge class >> onModel: aPCModel port: aPortNumber [

	^ self new
		model: aPCModel;
		port: aPortNumber;
		yourself
]

{ #category : 'accessing' }
CVBridge >> model: aPCModel [
	model := aPCModel
]

{ #category : 'accessing' }
CVBridge >> port: aPortNumber [
	port := aPortNumber
]

{ #category : 'accessing' }
CVBridge >> isRunning [
	^ server notNil
]

{ #category : 'actions' }
CVBridge >> start [

	server := CMGameServer new.
	server whenDataReceivedDo: [ :data :clientId |
		self handleFrame: data from: clientId ].
	server startOn: port.
]

{ #category : 'actions' }
CVBridge >> stop [

	server ifNil: [ ^ self ].
	server stop.
	server := nil.
]

{ #category : 'private' }
CVBridge >> handleFrame: data from: clientId [
	"Translate a CV detection frame into CORMAS model state.

	 Expected JSON shape (from demo.py / json_bridge.py):
	   { 'occupiedCells': [1, 5, 12, ...] }   -- 1-based cell ids where harvesters are detected
	"
	| occupiedCellIds harvesters |
	occupiedCellIds := data at: 'occupiedCells' ifAbsent: [ #() ].
	harvesters := model allInstances: PCHarvester.

	harvesters do: #leave.

	occupiedCellIds
		with: (harvesters first: occupiedCellIds size)
		do: [ :id :harvester |
			harvester moveTo: (model cellAt: id) ].
]
