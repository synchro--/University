/*ActiveThread viene usato sia dal client che dal server a seconda di chi è il pari 
 * attivo che richiede la connessione
 */

import java.net.*;
import java.io.*;

public class ActiveThread extends Thread {

private RemoteInfo remoteInfo; 
private String directory; 
private int mode; // mode = 0 per il client attivo mode = 1 per il server attivo

public ActiveThread(String directory, RemoteInfo remoteInfo, int mode)
{
	this.directory  = directory; 
	this.remoteInfo = remoteInfo; 
	this.mode = mode; 
}

public void run() {
 
 InetAddress addr  = null;
 addr = this.remoteInfo.getHost();
 int port = -1;
 port = this.remoteInfo.getPort();


 // oggetti utilizzati dal client per la comunicazione e la lettura del file
 // locale
 Socket socket = null;
 DataInputStream inSock = null;
 DataOutputStream outSock = null;


 try {
	 
	// sleep(500); //per sicurezza aspetto che il pari crei una socket di ascolto 
   // creazione socket e stream di input/output su socket, unica per tutta la
   // sessione
   try {
     socket = new Socket(addr, port);
     // setto il timeout 
     socket.setSoTimeout(30000);
     System.out.println("Creata la socket: " + socket);
     inSock = new DataInputStream(socket.getInputStream());
     outSock = new DataOutputStream(socket.getOutputStream());
   } 
   catch (IOException ioe) {
     System.out.println("GetFileActiveThread Problemi nella creazione degli stream su socket: ");
     ioe.printStackTrace();
     // il client esce in modo anomalo
     System.exit(1);
   }
   catch (Exception e) {
     System.out.println("Problemi nella creazione della socket: ");
     e.printStackTrace();
     // il client esce in modo anomalo
     System.exit(2);
   }

     
 File fileCorr = null;
 FileInfo[] fileToReceive = this.remoteInfo.getFileList(); 
     
     
	    
    
if(mode == 0)
{ //client attivo: quindi questa classe viene utilizzata dal client per ricevere i file
	 System.out.println("Client ricevo file..."); 

     socket.shutdownOutput(); // il client non deve inviare nulla
   
	try{
    	//creo il direttorio in cui salvare i file che mi invia il server
         File dir = new File(directory); 
    		if(dir.mkdir())
    		   System.out.println("Creato direttorio " + directory); 
    				 /*else {   System.out.print("Problemi nella creazione del direttorio " + nomeDirettorio); 
					          System.out.println("forse direttorio giÃ  esistente?"); }*/
    			     
    		    FileOutputStream outFileCorr;
    			 for(FileInfo fileInfo : fileToReceive) {
    			  System.out.println("Ricezione del file " + fileInfo.getFileName()
    			    	      + " di lunghezza " + fileInfo.getFileBytes() + " bytes");
    			    	      
    			    fileCorr = new File(dir.getName() + "/" +fileInfo.getFileName());
    			    long numBytes = fileInfo.getFileBytes(); 
    			    outfileCorr = new FileOutputStream(fileCorr);
    			    FileUtility.trasferisci_N_byte_file_binario(inSock, new DataOutputStream(outFileCorr),
    			    		numBytes);
    			}
	   }
   
    			 
      catch (EOFException eof) {
    			     System.out.println("Raggiunta la fine delle ricezioni, chiudo...");
    			     // e.printStackTrace();
    			     // finito il ciclo di ricezioni termino la comunicazione
    			     socket.close();
    			     // Esco con indicazione di successo
    			     System.out.println("ActiveThread: termino...");
    			   } 
    			   
    			   catch (SocketTimeoutException ste) {
    			     System.out.println("Timeout scattato: ");
    			     ste.printStackTrace();
    			     socket.close();
    			   } 
    			   
    			   catch (Exception e) {
    			     System.out.println("Problemi, i seguenti : ");
    			     e.printStackTrace();
    			     System.out.println("Chiudo ed esco...");
    			     socket.close();
    			   } 
}
    		
     //mode = 1 questa classe viene utilizzata dal server per inviare i file
    else if(mode == 1)
    {
 	  
       System.out.println("Server: invio file...");
       socket.shutdownInput(); //il server non deve ricevere nulla
 	    fileCorr = null; 
 	  
 	   try {
 	       File dirCorr = new File(directory);
 	       System.out.println("Apertura della directory " + directory);
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
    			     socket.close();
    			   }
    			   
 	     catch (Exception e) {
 	       System.out.println("Problemi nell'invio di " + fileCorr.getName()
 	           + ": ");
 	       e.printStackTrace();
 	       socket.close();
 	       // il client esce in modo anomalo
 	       System.exit(3);
 	     }
   } //else 

   // finita l'interazione chiudo la comunicazione col server
   socket.close();
   System.out.println("Chiusa comunicazione col pari.\nBye, bye!");

 }
 
 catch (Exception e) {
   System.err.println("Errore irreversibile, il seguente: ");
   e.printStackTrace();
   System.err.println("Chiudo!");
   System.exit(4);
  }
 } //run
} // ActiveThread
