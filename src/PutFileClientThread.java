//PutFileClient.java

import java.net.*;
import java.io.*;

public class PutFileClientThread extends Thread {

private RemoteInfo remoteInfo; 
private String directory; 

public PutFileClientThread(String directory, RemoteInfo remoteInfo)
{
	this.directory  = directory; 
	this.remoteInfo = remoteInfo; 
}

public void run() {
 /*
  * Come argomenti si devono passare un nome di host valido e una porta
  */
	
 InetAddress addr  = null;
try {
	addr = InetAddress.getByName(this.remoteInfo.getHost());
} catch (UnknownHostException e1) {
	
	e1.printStackTrace();
}

 int port = this.remoteInfo.getPort();


 // oggetti utilizzati dal client per la comunicazione e la lettura del file
 // locale
 Socket socket = null;
 DataInputStream inSock = null;
 DataOutputStream outSock = null;


 try {
   // creazione socket e stream di input/output su socket, unica per tutta la
   // sessione
   try {
     socket = new Socket(addr, port);
     // setto il timeout per non bloccare indefinitivamente il client
     socket.setSoTimeout(300000);
     System.out.println("Creata la socket: " + socket);
     inSock = new DataInputStream(socket.getInputStream());
     outSock = new DataOutputStream(socket.getOutputStream());
   } catch (IOException ioe) {
     System.out.println("Problemi nella creazione degli stream su socket: ");
     ioe.printStackTrace();
     System.out
         .print("\n^D(Unix)/^Z(Win)+invio per uscire, solo invio per continuare: ");
     // il client esce in modo anomalo
     System.exit(1);
   } catch (Exception e) {
     System.out.println("Problemi nella creazione della socket: ");
     e.printStackTrace();
     // il client esce in modo anomalo
     System.exit(2);
   }

     File fileCorr = null;
     FileInfo[] filesToReceive = this.remoteInfo.getFileList(); 
     
     
	     System.out.println("Client ricevo file..."); 
        
   try{
    	//creo il direttorio in cui salvare i file che mi invia† il server
    				
                     File dir = new File(directory); 
    				 if(dir.mkdir())
    					 System.out.println("Creato direttorio " + directory); 
    				 /*else {   System.out.print("Problemi nella creazione del direttorio " + nomeDirettorio); 
					          System.out.println("forse direttorio gi√† esistente?"); }*/
    				 
    				
//    				 String nomeFileRicevuto;
//    			     long numeroByte;
    			    
    			     // leggo il nome del file
//    			      
//    			       while ((nomeFileRicevuto = inSock.readUTF()) != null && !(nomeFileRicevuto.equalsIgnoreCase("fine"))) {
//    			    	 //String nomeFileCar = dir.getName() + "/";
//    			         fileCorr = new File(dir.getName() + "/" +nomeFileRicevuto);
//    			         if (fileCorr.exists()) {
//    			        	System.out.println("File " + fileCorr.getName() + " gia' presente, NON sovrascrivo");
//    			           outSock.writeUTF("No");
//    			           continue; //ricomincio il ciclo
//    			         }
//    			         
//    			         else {
//    			           outSock.writeUTF("attiva");
//    			           // leggo il numero di byte
//    			           numeroByte = inSock.readLong();
//    			           // controllo se il file esiste, se non esiste lo creo,
//    			           // altrimenti torno errore
//    			           System.out.println("Scrivo il file " + nomeFileRicevuto + " di "
//    			               + numeroByte + " byte");
//    			           outFileCorr = new FileOutputStream(fileCorr);
//    			           // trasferimento file
//    			           FileUtility.trasferisci_N_byte_file_binario(inSock,
//    			               new DataOutputStream(outFileCorr), numeroByte);
//    			           // chiusura file
//    			           outFileCorr.close();
//    			           }
//						  
//						  /****** NON RIESCE AD USCIRE DAL CICLO, NON LEGGE L'EOF BOH!! *******/  
//						 
//						 } // while
//					
//    	               /* quando la readUTF riceve un EOF lancia un eccezione
//    	               * che viene gestita chiudendo la comunicazione*/
//    			 }
    			     

    				 FileOutputStream outFileCorr;
    			     for(FileInfo fileInfo : filesToReceive)
    			     {
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
    			     socket.close();
    			     // Esco con indicazione di successo
    			     System.out.println("PutFileServer: termino...");
    			   } catch (SocketTimeoutException ste) {
    			     System.out.println("Timeout scattato: ");
    			     ste.printStackTrace();
    			     socket.close();
    			   } catch (Exception e) {
    			     System.out.println("Problemi, i seguenti : ");
    			     e.printStackTrace();
    			     System.out.println("Chiudo ed esco...");
    			     socket.close();
    			   } 
    		 //}//else 
		 
 
     
//     //caso put
//     else if(comando.equalsIgnoreCase("put"))
//     {
//     try {
//	  System.out.println("ricevuto comando put..."); 
//       // chiedo all'utente quale direttorio vuole inviare
//       System.out.print("\n Immetti nome direttorio: "); 
//       nomeDirettorio = stdIn.readLine(); 
//       File dirCorr = new File(nomeDirettorio);
//       String result;
//       if (dirCorr.exists() && dirCorr.isDirectory()) {
//    	   //invio nome direttorio 
//    	   outSock.writeUTF(nomeDirettorio); 
//    	   
//         File[] files = dirCorr.listFiles();
//         for (int i = 0; i < files.length; i++) {
//           fileCorr = files[i];
//           System.out.println("File con nome: " + fileCorr.getName());
//           if (fileCorr.isFile()) {
//             // Trasmissione: nome file
//             outSock.writeUTF(fileCorr.getName());
//             result = inSock.readUTF();
//             if (!result.equals("attiva")) System.out
//                 .println("Il file "
//                     + fileCorr.getName()
//                     + "era gia' presente sul server e non e' stato sovrascritto");
//             else {
//               System.out.println("Il file " + fileCorr.getName()
//                   + " NON e' presente sul server: inizio il trasferimento");
//               // lunghezza
//               outSock.writeLong(fileCorr.length());
//               // trasferimento dati
//               FileUtility.trasferisci_N_byte_file_binario(
//                   new DataInputStream(new FileInputStream(fileCorr
//                       .getAbsolutePath())), outSock, fileCorr.length());
//             }
//           }
//         }
//         // fine invio dei file nella cartella
//         System.out.print("\n^D(Unix)/^Z(Win)+invio per uscire, altrimenti immetti comando: ");
//		 continue; //ricomincia dall'inizio del while
//       } else {
//         System.out
//             .print(nomeDirettorio + " non e' un direttorio esistente");
//         System.out
//             .print("\n^D(Unix)/^Z(Win)+invio per uscire, altrimenti immetti comando: ");
//       }
//
//     }
//     /*
//      * NOTA: in caso di raggiungimento dell'EOF, la readUTF lancia una
//      * eccezione che viene gestita qui sotto chiudendo la socket e
//      * terminando il client con successo.
//      */
//     catch (EOFException e) {
//       System.out.println("Raggiunta la fine delle ricezioni, chiudo...");
//       // e.printStackTrace();
//       // finito il ciclo di ricezioni termino la comunicazione
//       socket.close();
//       // Esco con indicazione di successo
//       System.out.println("PutFileClient: termino...");
//       System.exit(0);
//     }
//      //altri errori
//     catch (Exception e) {
//       System.out.println("Problemi nell'invio di " + fileCorr.getName()
//           + ": ");
//       e.printStackTrace();
//       socket.close();
//       // il client esce in modo anomalo
//       System.exit(3);
//     }
//    } //else
			 

   // finita l'interazione chiudo la comunicazione col server
   socket.close();
   System.out.println("Chiusa comunicazione col server.\nBye, bye!");

 }
 // qui catturo le eccezioni non catturate all'interno del while
 // quali per esempio la caduta della connessione con il server
 // in seguito alle quali il client termina l'esecuzione
 catch (Exception e) {
   System.err.println("Errore irreversibile, il seguente: ");
   e.printStackTrace();
   System.err.println("Chiudo!");
   System.exit(4);
 }
} // main
} // PutFileClientThread
