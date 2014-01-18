//PutFileServer Concorrente

import java.io.*;
import java.net.*;

//Thread lanciato per ogni richiesta accettata
//versione per il trasferimento di file binari
class PutFileServerThread extends Thread {

private Socket clientSocket = null;
private String directory = null; 


// costruttore
public PutFileServerThread(Socket clientSocket, String directory) {
 this.clientSocket = clientSocket;
 this.directory = directory; 
}

/**
* Main invocabile con 0 o 1 argomenti. Argomento possibile -> porta su cui il
* server ascolta.
* 
*/
public void run() {
 DataInputStream inSock;
 DataOutputStream outSock;

 System.out.println("Attivazione figlio: "
     + Thread.currentThread().getName());
 try {
   try {
     // creazione stream di input e out da socket
     inSock = new DataInputStream(clientSocket.getInputStream());
     outSock = new DataOutputStream(clientSocket.getOutputStream());
   } catch (IOException ioe) {
     System.out
         .println("Problemi nella creazione degli stream di input/output "
             + "su socket: ");
     ioe.printStackTrace();
     // il server continua l'esecuzione riprendendo dall'inizio del ciclo
     return;
   } catch (Exception e) {
     System.out
         .println("Problemi nella creazione degli stream di input/output "
             + "su socket: ");
     e.printStackTrace();
     // il server continua l'esecuzione riprendendo dall'inizio del ciclo
     return;
   }

	   while(true){
	  
	  File fileCorr = null; 
	  
	   try {
	       File dirCorr = new File(directory);
	         File[] files = dirCorr.listFiles();
	         for (int i = 0; i < files.length; i++) {
	           fileCorr = files[i];
	           System.out.println("File con nome: " + fileCorr.getName());
	           if (fileCorr.isFile()) {
//	             // Trasmissione: nome file
//	             outSock.writeUTF(fileCorr.getName());
//	             result = inSock.readUTF();
//	             if (!result.equals("attiva")) System.out
//	                 .println("Il file "
//	                     + fileCorr.getName()
//	                     + "era gia' presente sul Client e non e' stato sovrascritto");
//	             else {
//	               System.out.println("Il file " + fileCorr.getName()
//	                   + " NON e' presente sul client: inizio il trasferimento");
	               // lunghezza
//	               outSock.writeLong(fileCorr.length());
	               // trasferimento dati
	               FileUtility.trasferisci_N_byte_file_binario(
	                   new DataInputStream(new FileInputStream(fileCorr
	                       .getAbsolutePath())), outSock, fileCorr.length());
	            // }
	           }
	         
	         // fine invio dei file nella cartella
	       }	  
		  }
	     /*
	      * NOTA: in caso di raggiungimento dell'EOF, la readUTF lancia una
	      * eccezione che viene gestita qui sotto chiudendo la socket e
	      * terminando il client con successo.
	      */
	     catch (EOFException e) {
	       System.out.println("Raggiunta la fine delle ricezioni, chiudo...");
	       // e.printStackTrace();
	       // finito il ciclo di ricezioni termino la comunicazione
	       clientSocket.close();
	       // Esco con indicazione di successo
	       System.out.println("PutFileServerThread: termino...");
	       System.exit(0);
	     }
	      //altri errori
	     catch (Exception e) {
	       System.out.println("Problemi nell'invio di " + fileCorr.getName()
	           + ": ");
	       e.printStackTrace();
	       clientSocket.close();
	       // il client esce in modo anomalo
	       System.exit(3);
	     }
    }
}
 catch (Exception e) {
   e.printStackTrace();
   // chiusura di stream e socket
   System.out
       .println("Errore irreversibile, PutFileServerThread: termino...");
 }
 
 System.out.println("Terminazione figlio: "
     + Thread.currentThread().getName());
} // run

} // PutFileServerThread

public class PutFileServerConThread extends Thread{
	
	private String directory; 
	private int port; 
	
	public PutFileServerConThread(String directory, int port)
	{
		this.directory = directory; 
		this.port = port;
	}

public void run(String[] args) throws IOException {


 ServerSocket serverSocket = null;
 Socket clientSocket = null;

 try {
   serverSocket = new ServerSocket(port);
   serverSocket.setReuseAddress(true); 
   System.out.println("PutFileServerConThread: avviato ");
   System.out.println("Server: creata la server socket: " + serverSocket);
 } catch (Exception e) {
   System.err
       .println("Server: problemi nella creazione della server socket: "
           + e.getMessage());
   e.printStackTrace();
   System.exit(1);
 }

 try {
	 

   while (true) {
     System.out.println("Server: in attesa di richieste...\n");

     try {
       // bloccante finche non avviene una connessione
       clientSocket = serverSocket.accept();
       clientSocket.setSoTimeout(60000);
       // anche se il server e' concorrente e' comunque meglio evitare che
       // un blocco indefinito
       System.out.println("Server: connessione accettata: " + clientSocket);
     } catch (Exception e) {
       System.err
           .println("Server: problemi nella accettazione della connessione: "
               + e.getMessage());
       e.printStackTrace();
       continue;
       // il server continua a fornire il servizio ricominciando dall'inizio
       // del ciclo
     }
     // servizio delegato ad un nuovo thread a cui passo anche la lista dei file da inviare
     try {
    	 
       //passo al thread la directory da aprire 	 
       new PutFileServerThread(clientSocket, directory).start();
       
     } catch (Exception e) {
       System.err.println("Server: problemi nel server thread: "
           + e.getMessage());
       e.printStackTrace();
       continue;
       // il server continua a fornire il servizio ricominciando dall'inizio
       // del ciclo
     }

   } // while
 }
 // qui catturo le eccezioni non catturate all'interno del while
 // in seguito alle quali il server termina l'esecuzione
 catch (Exception e) {
   e.printStackTrace();
   // chiusura di stream e socket
   System.out.println("PutFileServerCon: termino...");
   System.exit(2);
 }

}
} // PutFileServerConThread