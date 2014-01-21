
import java.io.File;
import java.rmi.Naming;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;



public class RMI_Server extends UnicastRemoteObject implements RMI_interfaceFile {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L; 
	static Prenotazione[] prenotazioni; 
	private static final int N = 5; 
        
        //costruttore
	public RMI_Server() throws RemoteException {
		super();
	}

             //implementazione metodi
             
	@Override
	public int elimina_prenotazione(String id) throws RemoteException {
		int res  = 2; //res = 0 ok res = 1 errore file res = 2 id sbagliato
		
		for(Prenotazione p: prenotazioni)
			if(p.getId().equals(id))
			{
				res = 1; 
				//elimino prima l'ImgFile della prenotazione 
				File f = new File(p.getNomeFile()); 
				if(f.delete()) res = 0; 
				//elimino la prenotazione ovvero setto tutti i valori a "L"
				p.reset(); 
					
			}
		
		return res; 
	}

	@Override
	public Prenotazione[] visualizza_prenotazione(String tipo, int numPers)
			throws RemoteException {
		
		Prenotazione[] res = null; 
		//vettore per tenere conto di quali sono da restituire e quali no 
		//in un solo ciclo
		boolean[] daRestituire = new boolean[N]; 
        for(boolean b : daRestituire) b =false; 
        
		int dim = 0; 
		
		for(int i=0; i < N; i++)
		{
			if(prenotazioni[i].getTipoPrenotazione().equals(tipo) 
					&& prenotazioni[i].getNumPersona() > numPers)
			{
				dim++; 
				daRestituire[i] = true; 
			}
		}
		
		res = new Prenotazione[dim]; 
		dim =0; //per riempire il vettore
		for(int i=0; i < N; i++)
		{
			if(daRestituire[i]) res[dim] = prenotazioni[i]; 
			
			dim++; 
		}
		
		return res;
	}
	
	
	private static void init()
	{
		prenotazioni = new Prenotazione[N]; 
		for(int i=0; i < N; i++)
			prenotazioni[i] = new Prenotazione(); 
		
		//valori di test 
		prenotazioni[0].setId("id"); 
		prenotazioni[0].setNumPersone(56);
		prenotazioni[0].setTipoPrenotazione("piazzola deluxe"); 
		prenotazioni[0].setVeicolo("niente"); 
		prenotazioni[0].setImgFile("ciao.txt"); 
	}
	

public static void main(String[] args)
{
	 int REGISTRYPORT = 1099; 
	 String registryHost = "localhost"; 
	 String serviceName = "server_RMI"; 
	 
	 //inizializzazione
	 init(); 
	 
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
	 }
	 catch(Exception e )
	 { System.out.println("Vi sono i seguenti errori: "); 
           e.printStackTrace(); 
           System.exit(-1);}
}



}
