
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.ArrayList;

public interface RMI_interfaceFile extends Remote {
	 
	//dichiarazione metodi
	
	//metodo da invocare all'inizio per inviare la lista delle directory presenti al client 
	//in modo che l'utente possa scegliere
	ArrayList<String> listDirectory() throws RemoteException; 
	
	//invio di lista file e endpoint server attivo
	RemoteInfo listFiles(String directory) throws RemoteException, UnknownHostException; 
	//invio di lista file client attivo (l'endpoint del client viene passato come parametro)
	RemoteInfo listFiles(String directory, InetAddress host, int port) throws RemoteException; 
	
	 

}