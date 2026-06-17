Class {
	#name : 'CMModelOpenerPlayCommand',
	#superclass : 'CMModelOpenerCommand',
	#category : 'Cormas-GameServer',
	#package : 'Cormas-GameServer'
}

{ #category : 'action' }
CMModelOpenerPlayCommand >> action [

	| presenter |
	presenter:=  CMGamePresenter
		forModelClass: owner cormasModelClass
		translator: translator.
			
	presenter open.
	presenter window centered
]

{ #category : 'accessing' }
CMModelOpenerPlayCommand >> icon [ 

	^ CMIcons playIcon
]

{ #category : 'testing' }
CMModelOpenerPlayCommand >> isBigButton [

	^ true
]

{ #category : 'accessing' }
CMModelOpenerPlayCommand >> name [

	^ 'Play'
]
