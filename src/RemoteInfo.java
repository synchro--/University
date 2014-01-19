import java.io.Serializable;

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
	private String host; 
	private int port; 
    
	private FileInfo[] remoteFiles; 
	
	public RemoteInfo(String host, int port, int numFiles)
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
	
	public String getHost()
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
