import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.UnknownHostException;
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
 		
      
      System.out.println("Scegli il metodo di scaricamento: "); 
      System.out.println("C = Client richiede la connessione " +
      		"S = server richiede la connessione");
			
    
      String nomeDir = null; 
      String service = null; 
      RemoteInfo fileList = null; 
      ArrayList<String> dirList = null; 
      int mode = -1; 
				
      
                //FINO A ESAURIMENTO INPUT UTENTE
		        while((service= stdIn.readLine())!=null)
                 {
                        if(service.equals("C")) //client attivo 
                        {
                        	 //ricevo la lista delle directory disponibili sul server 7
                        	dirList = new ArrayList<String>();
                        	dirList = serverRMI.listDirectory(); 
                            System.out.println("\nLe directory disponibili sul server sono: " + dirList);
                            System.out.println("Scegli la directory da scaricare, EOF per tornare " +
                            		" alla scelta tra client attivo/server attivo");
                            
                            //controllo che la directory scelta esista
                            nomeDir = stdIn.readLine(); 
                            while(nomeDir != null)
                         {
                            while(!dirList.contains(nomeDir))
                            {
                            	System.out.println("Scegli una directory corretta");
                            	 nomeDir = stdIn.readLine(); 
                            }
                            
                        	fileList = serverRMI.listFiles(nomeDir); //questa chiamata avvia il PutFileServerConThread
                        	//N.B. fileList contiene sia l'endpoint che le informazioni sui file	
                        	System.out.println("Server host: "+fileList.getHost().toString() + 
                        			"\tserver port: " + fileList.getPort());
                        	
//                        	System.out.println("Ecco la lista dei file");
//                        	System.out.println(fileList.getFileList().toString());
                        	
                        	mode = 0; //client attivo
                        	ActiveThread client = new ActiveThread(nomeDir,fileList,mode);
                        	client.start(); 
                        	Thread.sleep(5000); //attesa per le printf
                        	
                        	System.out.println("\nLe directory disponibili sul server sono: " + dirList);
                            System.out.println("Scegli la directory da scaricare, EOF per tornare " +
                            		" alla scelta tra client attivo/server attivo");
                        	nomeDir = stdIn.readLine();
                        	
                            } 
                        }

                        if(service.equals("S"))
                        {
                        	dirList = new ArrayList<String>();
                        	dirList = serverRMI.listDirectory(); 
                            System.out.println("\nLe directory disponibili sul server sono: " + dirList);
                            System.out.println("Scegli la directory da scaricare, EOF per tornare " +
                            		" alla scelta tra client attivo/server attivo");
                            //controllo che la directory scelta esista
                            nomeDir = stdIn.readLine(); 
                            while(nomeDir != null)
                         {
                            while(!dirList.contains(nomeDir))
                            {
                            	System.out.println("Scegli una directory corretta");
                            	 nomeDir = stdIn.readLine(); 
                            }
                             
                             InetAddress localHost = InetAddress.getByName(registryHost);
                             fileList = serverRMI.listFiles(nomeDir, localHost, 7868); //porta a caso
                             
                             System.out.println(fileList.getPort());
                            
                       	    mode = 1; //server attivo
                       	    PassiveConThread client = new PassiveConThread(nomeDir, fileList, mode);
                       	    client.start();
                       	    
                       	 	Thread.sleep(5000); //attesa per le printf
                            System.out.println("\nLe directory disponibili sul server sono: " + dirList);
                            System.out.println("Scegli la directory da scaricare, EOF per tornare " +
                            		" alla scelta tra client attivo/server attivo");
                            nomeDir = stdIn.readLine();
                            
                         }
                            
                        }
                        
                        else System.out.println("Servizio non disponibile"); 
                        
                        System.out.println("Scegli il metodo di scaricamento: "); 
                        System.out.println("C = Client richiede la connessione " +
                        		"S = server richiede la connessione");
                        
                       }
			
			
		}
		
		catch(UnknownHostException unk)
		{
			System.out.println("Host sconosciuto");
			unk.printStackTrace(); 
			System.exit(-2);
		}
		catch(Exception e )
		{
			System.out.println("Vi sono i seguenti errori: "); 
                        e.printStackTrace(); 
			System.exit(-1); 
		}
	}

}
