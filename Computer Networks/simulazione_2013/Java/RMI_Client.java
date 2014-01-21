import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.rmi.Naming;



public class RMI_Client {
	
	public static void main(String[] args)
	{
		int REGISTRYPORT = 1099; 
		String registryHost = "localhost"; 
		String serviceName = "server_RMI"; 
		
		  if (args.length != 0 && args.length != 1) {
		      System.out.println("Sintassi: RMI_Client [registryPort]");
		      System.exit(1);
		    }
		    if (args.length == 1) {
		      try {
		        REGISTRYPORT = Integer.parseInt(args[0]);
		      } catch (Exception e) {
		        System.out
		            .println("Sintassi: RMI_Client [registryPort], registryPort intero");
		        System.exit(2);
		      }
		    }
			 
		try{
			
			BufferedReader in = new BufferedReader(new InputStreamReader(System.in)); 
			String complete = "//" + registryHost + ":" + REGISTRYPORT + "/" + serviceName; 
			RMI_interfaceFile serverRMI = (RMI_interfaceFile) Naming.lookup(complete); 
			
			System.out.println("Client RMI: Servizio \"" + serviceName + "\" connesso");

      System.out.println("\nV=visualizza prenotazione\nE=elimina prenotazione\n EOF per terminare :");
			
                //FINO A ESAURIMENTO INPUT UTENTE
				String service = null; 
				String id = null; 
				String tipo = null; 
				int numPersone; 
				Prenotazione[] res = null; 
				
		        while((service = in.readLine())!=null)
                       {
                         if(service.equalsIgnoreCase("E"))
                         {
                        	 System.out.println("Inserisci id prenotazione"); 
                        	 id = in.readLine(); 
                        	 int esito = serverRMI.elimina_prenotazione(id); 
                        	 //controllo esito
                        	 if(esito == 0)
                        		 System.out.println("eliminazione effettuata correttamente");
                        	 if(esito == 1)
                        		 System.out.println("errore nell'eliminazione del file_IMg"); 
                        	 if(esito == 2)
                        		 System.out.println("Prenotazione non esistente"); 
                         }
                         
                         if(service.equalsIgnoreCase("V"))
                         {
                        	 do{
                        	 System.out.println("Inserisci tipo"); 
                        	 tipo = in.readLine(); 
                        	 } while(!tipo.equals("piazzola deluxe") && !tipo.equals("piazzola")
                        			 && !tipo.equals("mezza piazzola")); 
                        	 
                        	 System.out.println("Inserisci soglia persone"); 
                        	 try{
                        	 numPersone = Integer.parseInt(in.readLine()); 
                        	 }
                        	 catch(NumberFormatException e)
                        	 {
                        		 System.out.println("La soglia deve essere intera!"); 
                        		 continue; //il client continua a chiedere ciclicamente
                        	 }
                        	 
                        	 res = serverRMI.visualizza_prenotazione(tipo, numPersone);
                        	 for(Prenotazione p: res)
                        		 System.out.println(p.toString()); 	 
                        	 
                         }

             System.out.println("\nV=visualizza prenotazione\nE=elimina prenotazione\n EOF per terminare :"); 
                            
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