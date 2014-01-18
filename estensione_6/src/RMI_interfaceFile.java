
import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.ArrayList;

public interface RMI_interfaceFile extends Remote {
	 
	//dichiarazione metodi
	
	//metodo da invocare all'inizio per inviare la lista delle directory presenti al client 
	//in modo che l'utente possa scegliere
	ArrayList<String> listDirectory() throws RemoteException; 
	
	//invio di lista file e endpoint
	RemoteInfo listFiles(String directory) throws RemoteException; 
	
	 

}