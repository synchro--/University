
import java.io.File;
import java.rmi.Naming;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.ArrayList;



public class RMI_Server extends UnicastRemoteObject implements RMI_interfaceFile {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
        
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
public RemoteInfo listFiles(String directory) throws RemoteException {
	
	return null;
}
	
	
	

public static void main(String[] args)
{
	 int REGISTRYPORT = 1099; 
	 String registryHost = "localhost"; 
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
