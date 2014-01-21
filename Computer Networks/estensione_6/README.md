This is a client-server application based on Java Remote Method Invocation (RMI)and socket both. The client asks the server the list of remote directories and ask the client to choose the directory to download. Then the client send the chosen directory name to server and the server returns the list of files names and files lengths and his endpoint. All of these is performed using RMI.
Afterthat the C/S communication goes on with stream sockets and the files are sent from server to client. 

Fileinfo contains the name and length of 1 file.  
RemoteInfo contains the endpoint useful for socket stream and an array of FileInfo. 
[still incomplete..]

Italiano: 
Le classi RemoteInfo e FileInfo potrebbero essere superflue (si potrebbero inviare i file tramite un protocollo con il pari vedi esercitazione 2) però così seguono tutte le specifiche dell'estensione. 

Classi: 
FileInfo contiene le informazioni relative ad un file: nome e lunghezza in bytes. 
RemoteInfo contiene l'endpoint del client/server a seconda di chi fa il pari attivo e un array di FileInfo contenente tutti i nomi e tutte le lunghezze
           di tutti i file che devono essere inviati 
		   
ActiveThread: Thread invocato sia da client che da server a seconda di chi deve richiedere la connessione in modo attivo. Questo avviene per mezzo di una variabile mode che discrimina tra client attivo (mode=0) e quindi ciclo di ricezione file o server attivo(mode=1) ciclo di invio file. Stessa cosa succede per il PassiveThreadCon. Utilizza l'oggetto RemoteInfo passato come parametro per reperire l'endpoint del pari per connettersi ed eventualmente la lista di file da ricevere se mode=0 (client attivo). 

PassiveThreadCon: viene utilizzato dal pari che si mette in ascolto. E un thread concorrente che una volta ricevuta la richiesta genera un altro thread per eseguirla. Siccome viene eseguito nuovamente ogni volta che si invoca un metodo RMI si poteva implementarlo in modo sequenziale ma ho voluto seguire la struttura a thread concorrente di un classico server che si mette in ascolto. (vedi esercitazione n.2)
