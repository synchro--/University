
import java.io.File;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.rmi.Naming;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.ArrayList;



public class RMI_Server extends UnicastRemoteObject implements RMI_interfaceFile {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private static int REGISTRYPORT; 
	private static String registryHost; 
        
        //costruttore
	public RMI_Server() throws RemoteException {
		super();
	}

	
             //implementazione metodi
		
@Override
public ArrayList<String> listDirectory() throws RemoteException {
	
	ArrayList<String> res = new ArrayList<String>();
	
	File dir = new File("."); 
	File[] list = dir.listFiles(); 
	for(File file : list)
	{
		if(file.isDirectory())
			res.add(file.getName());
	}
	
	return res; 
}

@Override
public RemoteInfo listFiles(String directory) throws RemoteException, UnknownHostException {
	
	String currentNameFile = null; 
	long currentFileBytes = -1; 
	File dir = new File(directory); 
	File[] list = null;
	         
	list  = dir.listFiles();
	InetAddress host = InetAddress.getByName(registryHost);
	RemoteInfo res = new RemoteInfo(host, REGISTRYPORT, list.length);
	
	for(File f: list)
	{
		//se la directory contiene un'altra directory questa non viene inclusa
		if(f.isFile())
		{
			currentNameFile = f.getName(); 
			currentFileBytes = f.length(); 
			if(!res.addFile(new FileInfo(currentNameFile, currentFileBytes)))
			 System.err.println("Errore nel completamento della lista dei file");
		}
	}
	
	System.out.println("Avvio il Thread server");
	//creazione socket 
	int mode = 0; //client attivo
	GetFileServerConThread server = new GetFileServerConThread(directory,res,mode);	
	server.start();
	
	System.out.println("Inviato l'endpoint al Client");	
	return res; 
}


@Override
public RemoteInfo listFiles(String directory, InetAddress host, int port)
		throws RemoteException {
	String currentNameFile = null; 
	long currentFileBytes = -1; 
	File dir = new File(directory); 
	File[] list = null;
	         
	list  = dir.listFiles();
	
	/* In questo caso dobbiamo inviare al client solo una lista di File. Per non dichiarare altre variabili
	 * utilizzeremo sempre un oggetto RemoteInfo che contiene la lista + endpoint del client.
	 * L'endpoint del client viene ignorato dal client (a cui interessa solo la lista) ma viene invece
	 * utilizzato dal server in modalita' attiva
	 */
	
	RemoteInfo res = new RemoteInfo(host, port, list.length);
	
	for(File f: list)
	{
		//se la directory contiene un'altra directory questa non viene inclusa
		if(f.isFile())
		{
			currentNameFile = f.getName(); 
			currentFileBytes = f.length(); 
			if(!res.addFile(new FileInfo(currentNameFile, currentFileBytes)))
			 System.err.println("Errore nel completamento della lista dei file");
		}
	}
	
	/* Per avere un server attivo anziche' creare un'altra classe identica a GetFileClientThread 
	 * utilizziamo questa ma con valore discriminante 1
	 */
	
	int mode = 1; //server attivo
	GetFileClientThread server = new GetFileClientThread(directory, res, mode);
	server.start();
	
	return res;
}

	
	

public static void main(String[] args)
{
	  REGISTRYPORT = 1099; 
	  registryHost = "localhost"; 
	 String serviceName = "server"; 
	 
	 // Controllo dei parametri della riga di comando
    if (args.length != 0 && args.length != 1) {
      System.out.println("Sintassi: RMI_Server [registryPort]");
      System.exit(1);
    }
    if (args.length == 1) {
      try {
        REGISTRYPORT = Integer.parseInt(args[0]);
      } catch (Exception e) {
        System.out
            .println("Sintassi: RMI_Server [registryPort], registryPort intero");
        System.exit(2);
      }
      
    }
	 
	 
	 try{
		 String complete = "//"+registryHost+":"+ REGISTRYPORT + "/" + serviceName; 
		 
		 RMI_Server serverRMI = new RMI_Server(); 
		 
		 Naming.rebind(complete, serverRMI); 
		 
		 System.out.println("Server avviato");
		 
		 
		 
	 }
	 catch(Exception e )
	 { System.out.println("Vi sono i seguenti errori: "); 
           e.printStackTrace(); 
           System.exit(-1);}
}


}
