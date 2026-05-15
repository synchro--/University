# TODO 

### MNIST 
- provare fine-tuning su modello mnist classico con primo layer non approssimato già come architettura 
- provare ad aggiungere dei BN tra i 4 layer di convoluzione e provare il retrain 
- provare un approssimazione migliore del tensore CPD
- provare stessa architettura ma con FC sostituiti da quelli di convoluzione...  
- provare a fare fine-tuning con pesi random ma stessa dimensione di MNIST che da 99% di accuracy OK
- provare fine-tuning senza freeze: comunque non funziona, risultati pessimi OK 
- provare a ad usare i SeparableConv2D nel mezzo degli altri due layer iniziali
- 

### CIFAR 
- provare con la rete di CIFAR10: prima singolo layer con fine-tuning totale. Poi secondo layer con fine-tuning totale. OK 
- provare nuova architettura CIFAR10: Accuracy only 60% senza batch normalization 
- provare nuova architettura con BN: Accuracy aumentata rispetto al modello originale: 79-> 82% e molti meno params! 
- provare a fare Fine-Tuning con solo pesi random: pessimi risultati, quindi può essere che con cifar il trucchetto non funziona? 
- provare a fare Fine-Tuning con pesi random in 1 solo layer, ma sul modello con BN e architettura trainata già modulare


### New Architecture 
- provare ad usare i SeparableConv2D e vedere se cambia qualcosa 
- provare a sostituire i layer finali con FC e vedere quanti parametri cambiano sul modello originale 
- provare anche con VGG ed AlexNet fare fine-tuning 

---
- inserire un layer con pesi custom e testare la rete su MNIST di nuovo con risultati chiaramente pessimi OK 
- salvare i pesi di un layer della rete in formato .mat OK 
- fare la CPD di questo layer e salvarlo sempre in .mat OK
- ricaricare questo layer in python e inserirlo nella rete come nel punto 1 OK
- modificare quindi le dimensioni richieste dal layer e vedere i risultati OK
- fare fine-tuning di questo layer OK
- Fare passo 2) con Matlab: 
	- banco di filtri conosciuti, magari anche lo stesso (tipo sobel..) 
	- immagine in ingresso --> output 4 immagini identiche 
	- CPD del banco di matrici (tensore di filtri conosciuti) 
	- conv con il CPD ottenuto e confronto risultati visivamente
	- "" "" confronto risultati con STNR, RBMS, ecc. 



### Dal Paper 
- capire se i pesi del tensore a 4 `[d, d, in, out]` sono assegnati in maniera corretta dalle matrici a due dimensioni `[X, Rank]` ottenute da CPD 
- su quali CNN provarlo? Alexnet... 
- freeze o non freeze degli altri layer? 
