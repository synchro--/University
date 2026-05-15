
import java.rmi.Remote;
import java.rmi.RemoteException;

public interface RMI_interfaceFile extends Remote {
	 
         //dichiarazione metodi
	 public int elimina_prenotazione(String id) throws RemoteException; 
	 
	 public Prenotazione[] visualizza_prenotazione(String tipo, int numPers) throws RemoteException; 

}
