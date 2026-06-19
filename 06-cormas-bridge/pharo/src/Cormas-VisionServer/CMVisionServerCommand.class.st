Class {
	#name : 'CMVisionServerCommand',
	#superclass : 'CMSpacePresenterToggleCommand',
	#classInstVars : [
		'server'
	],
	#category : 'Cormas-VisionServer',
	#package : 'Cormas-VisionServer'
}

{ #category : 'accessing' }
CMVisionServerCommand class >> server [

	^ server ifNil: [ server := CMGameServer new ] 
]

{ #category : 'accessing' }
CMVisionServerCommand >> actionWhenOff [

	self class server stop
]

{ #category : 'accessing' }
CMVisionServerCommand >> actionWhenOn [

	self class server startOn: 8081
]

{ #category : 'accessing' }
CMVisionServerCommand >> iconWhenOff [

	^ CMIcons runIcon
]

{ #category : 'accessing' }
CMVisionServerCommand >> iconWhenOn [
	^self iconNamed: #cancel
]

{ #category : 'initialization' }
CMVisionServerCommand >> initialize [

	super initialize.

	self class server whenDataReceivedDo: [ :data :clientId |
		| occupiedCellIds biomassCellIds birdsCellIds protectedCellIds harvesters model allCells |
		model := self owner cormasModel.
		occupiedCellIds  := data at: 'occupiedCells' ifAbsent: [ #() ].
		biomassCellIds   := data at: 'biomass'       ifAbsent: [ #() ].
		birdsCellIds     := data at: 'birds'         ifAbsent: [ #() ].
		protectedCellIds := data at: 'protected'     ifAbsent: [ #() ].

		harvesters := model @@ PCHarvester.
		harvesters do: #leave.
		occupiedCellIds
			with: (harvesters first: occupiedCellIds size)
			do: [ :id :harvester | harvester moveTo: (model cellAt: id) ].

		allCells := model @@ PCCell.
		allCells do: [ :cell | cell biomass: 0 ].
		biomassCellIds do: [ :id | (model cellAt: id) biomass: 3 ].

		allCells do: [ :cell | cell numberOfNewborns: 0 ].
		birdsCellIds do: [ :id | (model cellAt: id) numberOfNewborns: 2 ].

		allCells do: [ :cell | cell makeNotProtected ].
		protectedCellIds do: [ :id | (model cellAt: id) makeProtected ].

		self owner updateDiagram ]
]

{ #category : 'accessing' }
CMVisionServerCommand >> nameWhenOff [ 
	^'Start Server'
]

{ #category : 'accessing' }
CMVisionServerCommand >> nameWhenOn [ 
	^'Stop Server'
]

{ #category : 'accessing' }
CMVisionServerCommand >> owner [
	^ owner
]
