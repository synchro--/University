/*Thread passivo Concorrente che rimane in ascolto utilizzando ServerSocket e quando accetta 
 * una connessione genera un figlio per soddisfare la richiesta. Utilizzato dal pari si mette in ascolto.
 * N.B: Si sarebbe potuto creare un unico thread che riceve le richiesta e le attua in modo sequenziale
 * pero' ho preferito mantenere la stessa struttura dell'esercitazione n.2 
 */

import java.io.*;
import java.net.*;

//Thread lanciato dal PassiveConThread per ogni richiesta accettata
//versione per il trasferimento di file binari
class PassiveThread extends Thread {

private Socket clientSocket = null;
private String directory = null; 
private FileInfo[] list = null; //utilizzato in mode = 1 per server attivo
private int mode; // vedi sotto


// costruttore
public PassiveThread(Socket clientSocket, String directory, FileInfo[] list, int mode) {
 this.clientSocket = clientSocket;
 this.directory = directory; 
 this.list = list; 
 this.mode = mode;
}


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
   
   File fileCorr = null; 

   if(mode == 0)
   {
	   try {
	       File dirCorr = new File(directory);
	       File[] files = dirCorr.listFiles();
	       for (int i = 0; i < files.length; i++) {
	           fileCorr = files[i];
	           System.out.println("File con nome: " + fileCorr.getName());
	           if (fileCorr.isFile()) {
	               FileUtility.trasferisci_N_byte_file_binario(
	                   new DataInputStream(new FileInputStream(fileCorr
	                       .getAbsolutePath())), outSock, fileCorr.length());
	           }
	         
	         // fine invio dei file nella cartella
	       }	  
		  }
	     catch (SocketTimeoutException ste) {
	    			     System.out.println("Timeout scattato: ");
	    			     ste.printStackTrace();
	    			     clientSocket.close();
	    			   }
	    			   
	     catch (Exception e) {
	       System.out.println("Problemi nell'invio di " + fileCorr.getName()
	           + ": ");
	       e.printStackTrace();
	       clientSocket.close();
	       // esce in modo anomalo
	       System.exit(3);
	     }
    }
   
   else if(mode == 1) //UTILIZZATO DAL CLIENT   
   {

		  try{
	    	//creo il direttorio in cui salvare i file che mi invia il server
	                     File dir = new File(directory); 
	    				 if(dir.mkdir())
	    					 System.out.println("Creato direttorio " + directory); 
	    			 
	    				 FileOutputStream outFileCorr;
	    			  for(FileInfo fileInfo : this.list)
	    			 {
	    			    	 System.out.println("Ricezione del file " + fileInfo.getFileName()
	    			    			 + " di lunghezza " + fileInfo.getFileBytes() + " bytes");
	    			    	 fileCorr = new File(dir.getName() + "/" +fileInfo.getFileName());
	    			    	 long numBytes = fileInfo.getFileBytes(); 
	    			    	 outFileCorr = new FileOutputStream(fileCorr);
	    			    	 FileUtility.trasferisci_N_byte_file_binario(inSock, new DataOutputStream(outFileCorr),
	    			    		numBytes);
	    			 }
		   }
	   
	    			 catch (EOFException eof) {
	    			     System.out.println("Raggiunta la fine delle ricezioni, chiudo...");
	    			     // e.printStackTrace();
	    			     // finito il ciclo di ricezioni termino la comunicazione
	    			     clientSocket.close();
	    			     // Esco con indicazione di successo
	    			     System.out.println("PassiveThread: termino...");
	    			   } catch (SocketTimeoutException ste) {
	    			     System.out.println("Timeout scattato: ");
	    			     ste.printStackTrace();
	    			     clientSocket.close();
	    			   } catch (Exception e) {
	    			     System.out.println("Problemi, i seguenti : ");
	    			     e.printStackTrace();
	    			     System.out.println("Chiudo ed esco...");
	    			     clientSocket.close();
	    			   } 
   }
}
 catch (Exception e) {
   e.printStackTrace();
   // chiusura di stream e socket
   System.out
       .println("Errore irreversibile, PassiveThread: termino...");
 }
 
 System.out.println("Terminazione figlio: "
     + Thread.currentThread().getName());
 
} // run

} // PassiveThread




public class PassiveConThread extends Thread{
	
	private String directory; 
	private RemoteInfo remoteInfo; 
 private int mode; /*se mode = 0 allora client attivo --> il server usa questa classe per inviare file
                      se mode = 1 allora server attivo --> il client usa questa classe per ricevere file*/
	
	public PassiveConThread(String directory, RemoteInfo remoteInfo, int mode)
	{
		this.directory = directory; 
		this.remoteInfo = remoteInfo; 
		this.mode = mode; 
	}

public void run() {


 ServerSocket serverSocket = null;
 Socket clientSocket = null;
 int port = this.remoteInfo.getPort(); 
 //int port = 6666; 

 try {
   serverSocket = new ServerSocket(port);
   serverSocket.setReuseAddress(true); 
   System.out.println("PassiveConThread: avviato ");
   System.out.println("Server: creata la server socket: " + serverSocket);
 } catch (Exception e) {
   System.err
       .println("Server: problemi nella creazione della server socket: "
           + e.getMessage());
   e.printStackTrace();
   System.exit(1);
 }

 try {
     System.out.println("Server: in attesa di richieste...\n");

     try {
       // bloccante finche non avviene una connessione
       clientSocket = serverSocket.accept();
       clientSocket.setSoTimeout(60000);
       // anche se e' concorrente e' comunque meglio evitare un blocco indefinito
       System.out.println("PassiveThreadCon: connessione accettata: " + clientSocket);
     } catch (Exception e) {
       System.err
           .println("PassiveThreadCon: problemi nella accettazione della connessione: "
               + e.getMessage());
       e.printStackTrace();
     }
     // servizio delegato ad un nuovo thread
     try {
    	 
       //passo al thread la directory da aprire e la lista dei file in caso di mode = 1
       FileInfo[] list = this.remoteInfo.getFileList();
       new PassiveThread(clientSocket, directory,list,mode).start();
       
     } catch (Exception e) {
       System.err.println("PassiveThreadCon: problemi nel server thread: "
           + e.getMessage());
       e.printStackTrace();
     }
 }
 // qui catturo le eccezioni non catturate all'interno del while
 // in seguito alle quali il server termina l'esecuzione
 catch (Exception e) {
   e.printStackTrace();
   // chiusura di stream e socket
   System.out.println("PassiveThreadCon: termino...");
   System.exit(2);
 }

try {
	serverSocket.close();
  } catch (IOException e) {
	System.out.println("Errore nella chiusura server socket");
	e.printStackTrace();
 } 

 }
} // PassiveConThread