Class {
	#name : 'CMGamePresenter',
	#superclass : 'SpPresenter',
	#instVars : [
		'translator',
		'modelClass',
		'initializer',
		'simulationRunner'
	],
	#category : 'Cormas-GameServer',
	#package : 'Cormas-GameServer'
}

{ #category : 'examples' }
CMGamePresenter class >> example [
	<example>
	(self forModelClass: CMMockModel translator: CMEnglishTranslator new) open
]

{ #category : 'instance creation' }
CMGamePresenter class >> forModelClass: aClass translator: aTranslator [

	^ self on: { aClass . aTranslator }
]

{ #category : 'as yet unclassified' }
CMGamePresenter >> beInNotRunningMode [

	"do nothing"
]

{ #category : 'initialization' }
CMGamePresenter >> connectPresenters [

	initializer
		whenInitializedDo: [ simulationRunner simulation: initializer simulation ];
		beReinitializeCondition: [ simulationRunner hasSimulationData ].
]

{ #category : 'layout' }
CMGamePresenter >> defaultLayout [

	^ SpBoxLayout newLeftToRight
		spacing: 10;
		add: initializer width: 250;
		add: simulationRunner;
		yourself
]

{ #category : 'initialization' }
CMGamePresenter >> initializePresenters [
	
	initializer := self
		instantiate: CMSimulationInitializerPresenter
		on: { modelClass . translator }.
		
	simulationRunner := self
		instantiate: CMSimulationRunnerPresenter
		on: translator.
]

{ #category : 'initialization' }
CMGamePresenter >> initializeWindow: aWindowPresenter [

	super initializeWindow: aWindowPresenter.

	aWindowPresenter 
		title: (modelClass modelName, ' - Game');
		initialExtent: 1000@600;
		whenResizingDo: [ simulationRunner fitDiagram ].
]

{ #category : 'accessing - model' }
CMGamePresenter >> setModelBeforeInitialization: aCollection [

	modelClass := aCollection first.
	translator := aCollection second
]
