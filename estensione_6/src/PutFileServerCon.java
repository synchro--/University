//PutFileServer Concorrente

import java.io.*;
import java.net.*;

//Thread lanciato per ogni richiesta accettata
//versione per il trasferimento di file binari
class PutFileServerThread extends Thread {

private Socket clientSocket = null;

// costruttore
public PutFileServerThread(Socket clientSocket) {
 this.clientSocket = clientSocket;
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
   
   
   if(inSock.readUTF().equalsIgnoreCase("get"))
   {
	  //invio lista cartelle disponibili 
	  File dir = new File(".");
	  File[] lista = dir.listFiles(); 
	  for(File f : lista)
		  if(f.isDirectory())
			  outSock.writeUTF(f.getName()); 
	  
	  String nomeDirettorio = null; 
	  File fileCorr = null; 
	  
	   try {
	       // Se il direttorio esiste allora invio al client tutti i file 
		   //altrimenti invio una stringa "non esiste" 
	       
	       nomeDirettorio = inSock.readUTF(); 
	       File dirCorr = new File(nomeDirettorio);
	       String result;
	       if (dirCorr.exists() && dirCorr.isDirectory()) {
	         File[] files = dirCorr.listFiles();
	         for (int i = 0; i < files.length; i++) {
	           fileCorr = files[i];
	           System.out.println("File con nome: " + fileCorr.getName());
	           if (fileCorr.isFile()) {
	             // Trasmissione: nome file
	             outSock.writeUTF(fileCorr.getName());
	             result = inSock.readUTF();
	             if (!result.equals("attiva")) System.out
	                 .println("Il file "
	                     + fileCorr.getName()
	                     + "era gia' presente sul Client e non e' stato sovrascritto");
	             else {
	               System.out.println("Il file " + fileCorr.getName()
	                   + " NON e' presente sul client: inizio il trasferimento");
	               // lunghezza
	               outSock.writeLong(fileCorr.length());
	               // trasferimento dati
	               FileUtility.trasferisci_N_byte_file_binario(
	                   new DataInputStream(new FileInputStream(fileCorr
	                       .getAbsolutePath())), outSock, fileCorr.length());
	             }
	           }
	         }
	         // fine invio dei file nella cartella
	       } else {
	    	   result = "non esiste";
	          outSock.writeUTF(result);
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
	       System.out.println("PutFileServer: termino...");
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
   
   //caso put
   else
   {
	 try{  
     String nomeFileRicevuto;
     long numeroByte;
     File fileCorr;
     FileOutputStream outFileCorr;
     // leggo il nome del file
       while ((nomeFileRicevuto = inSock.readUTF()) != null) {
         fileCorr = new File(nomeFileRicevuto);
         if (fileCorr.exists()) {
           outSock.writeUTF("File gia' presente, NON sovrascrivo");
         } else {
           outSock.writeUTF("attiva");
           // leggo il numero di byte
           numeroByte = inSock.readLong();
           // controllo se il file esiste, se non esiste lo creo,
           // altrimenti
           // torno errore
           System.out.println("Scrivo il file " + nomeFileRicevuto + " di "
               + numeroByte + " byte");
           outFileCorr = new FileOutputStream(nomeFileRicevuto);
           // trasferimento file
           FileUtility.trasferisci_N_byte_file_binario(inSock,
               new DataOutputStream(outFileCorr), numeroByte);
           // chiusura file
           outFileCorr.close();
         }
       } // while

     /*
      * NOTA: in caso di raggiungimento dell'EOF, la readUTF lancia una
      * eccezione che viene gestita qui sotto chiudendo la socket e
      * terminando il client con successo.
      */
   } catch (EOFException eof) {
     System.out.println("Raggiunta la fine delle ricezioni, chiudo...");
     // e.printStackTrace();
     // finito il ciclo di ricezioni termino la comunicazione
     clientSocket.close();
     // Esco con indicazione di successo
     System.out.println("PutFileServer: termino...");
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
  }//else
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

public class PutFileServerCon {

public static void main(String[] args) throws IOException {

 // Porta sulla quale ascolta il server
 int port = -1;

 /* controllo argomenti */
 try {
   if (args.length == 1) {
     port = Integer.parseInt(args[0]);
     // controllo che la porta sia nel range consentito 1024-65535
     if (port < 1024 || port > 65535) {
       System.out.println("Usage: java LineServer [serverPort>1024]");
       System.exit(1);
     }
   } else {
     System.out.println("Usage: java PutFileServerCon port");
     System.exit(1);
   }
 } // try
 catch (Exception e) {
   System.out.println("Problemi, i seguenti: ");
   e.printStackTrace();
   System.out.println("Usage: java PutFileServerCon port");
   System.exit(1);
 }

 ServerSocket serverSocket = null;
 Socket clientSocket = null;

 try {
   serverSocket = new ServerSocket(port);
   serverSocket.setReuseAddress(true); 
   System.out.println("PutFileServerCon: avviato ");
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

     // serizio delegato ad un nuovo thread
     try {
       new PutFileServerThread(clientSocket).start();
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
} // PutFileServerCon