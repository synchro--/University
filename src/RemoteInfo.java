import java.io.Serializable;
import java.net.InetAddress;

/*struttura che contiene l'endpoint 
 * e la lista dei file con le rispettive lunghezze 
 * da passare al client 
 */


public class RemoteInfo implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	//endpoint
	private InetAddress host; 
	private int port; 
    
	private FileInfo[] remoteFiles; 
	
	public RemoteInfo(InetAddress host, int port, int numFiles)
	{
		this.host = host; 
		this.port = port;
		
		this.remoteFiles = new FileInfo[numFiles]; 
		
		//init
		for(int i=0; i < numFiles; i++)
		{
			remoteFiles[i] = new FileInfo();
		}
		
	}
	
	public boolean addFile(FileInfo f)
	{
		for(int i=0; i < remoteFiles.length; i++)
		{
			//se libera la riempio
			if(remoteFiles[i].getFileName().equals("L"))
			{
			    remoteFiles[i] = f;
			    return true; 
			}
		}
		
		return false; 
	}
	
	public InetAddress getHost()
	{
		return host; 
	}
	
	public int getPort()
	{
		return port; 
	}
	
	public FileInfo[] getFileList()
	{
		return remoteFiles; 
	}
	
	
}
