import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.rmi.Naming;
import java.util.ArrayList;



public class RMI_Client {
	
	public static void main(String[] args)
	{
		int REGISTRYPORT = 1099; 
        String registryHost = null;
        String serviceName = "server";
        BufferedReader stdIn = new BufferedReader(new InputStreamReader(System.in));

    // Controllo dei parametri della riga di comando
    if (args.length != 1 && args.length != 2) {
      System.out.println("Sintassi: ClientFile NomeHost [registryPort]");
      System.exit(1);
    } else {
      registryHost = args[0];
      if (args.length == 2) {
        try {
          REGISTRYPORT = Integer.parseInt(args[1]);
        } catch (Exception e) {
          System.out
              .println("Sintassi: ClientFile NomeHost [registryPort], registryPort intero");
          System.exit(2);
        }
      }
    }
		
		try{
			
			String complete = "//" + registryHost + ":" + REGISTRYPORT + "/" + serviceName; 
			RMI_interfaceFile serverRMI = (RMI_interfaceFile) Naming.lookup(complete); 
			
			System.out.println("Client RMI: Servizio \"" + serviceName + "\" connesso");
      
	 //ricevo la lista delle directory disponibili sul server 
	 ArrayList<String> dirList = new ArrayList<String>(); 
	 
	 dirList = serverRMI.listDirectory(); 
			
      System.out.println("\nLe directory disponibili sul server sono: " + dirList);
      System.out.println("Scegli la directory da scaricare");
			
      //FINO A ESAURIMENTO INPUT UTENTE
      String nomeDir = null; 
      String service = null; 
      RemoteInfo fileList = null; 
				
		        while((nomeDir= stdIn.readLine())!=null)
                 {
                        System.out.println("Scegli il metodo di scaricamento: "); 
                        System.out.println("C = Client richiede la connessione " +
                        		"S = server richiede la connessione");
                        service=stdIn.readLine(); 
                        if(service.equals("C"))
                        {
                        	fileList = serverRMI.listFiles(nomeDir);
                            
                        }

                            
                       }
			
			
		}
		
		catch(Exception e )
		{
			System.out.println("Vi sono i seguenti errori: "); 
                        e.printStackTrace(); 
			System.exit(-1); 
		}
	}

}
