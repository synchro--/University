import java.io.Serializable;


public class FileInfo implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private String fileName; 
	private long bytes; 
	
	public FileInfo()
	{
		this.fileName = "L"; 
		this.bytes = -1; 
	}
	
	public FileInfo(String nomeFile, long bytes)
	{
		this.fileName = nomeFile; 
		this.bytes = bytes; 
	}
	
	public String getFileName()
	{
		return fileName; 
	}
	
	public long getFileBytes()
	{
		return bytes; 
	}
	
	

}
