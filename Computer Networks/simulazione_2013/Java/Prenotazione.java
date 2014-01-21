//package simulazione2013;

import java.io.Serializable;

public class Prenotazione implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private String id; 
	private int numPersona; 
	private String tipoPren; 
	private String veicolo; 
	private String targa; 
	private String imgFile; 
	
	public Prenotazione()
	{
		this.numPersona = -1; 
		this.id = this.tipoPren = this.veicolo = this.targa = this.imgFile = "L"; 
		 
	}
	
	//metodi setter & getter (solo quelli relativi alle funzionalità da implementare)
	
	//in questo caso servono solo delle get... 
	public String getId()
	{
		return id; 
	}
	
	public int getNumPersona()
	{
		return numPersona; 
	}
	
	public String getTipoPrenotazione()
	{
		return this.tipoPren; 
	}

	public String getVeicolo()
	{
		return veicolo; 
	}
	
	public String getNomeFile()
	{
		return imgFile; 
	}
	
	public synchronized void reset()
	{
		this.id = this.imgFile = this.tipoPren 
		= this.targa = this.veicolo = "L"; 
		
		this.numPersona = -1; 
		
		
	}
	
	public synchronized void setId(String id)
	{
		this.id = id;
	}
	
	public synchronized void setNumPersone(int n)
	{
		this.numPersona = n; 
	}
	
	public synchronized void setTipoPrenotazione(String tipo)
	{
		this.tipoPren = tipo; 
	}
	
	public synchronized void setVeicolo(String veicolo)
	{
		this.veicolo = veicolo; 
	}
	
	public synchronized void setImgFile(String file)
	{
		this.imgFile = file; 
	}
	public String toString()
	{
		return id + "  " + this.numPersona + "  " + this.tipoPren + "  " + 
	      this.veicolo + "  " + this.targa +  "  " + this.imgFile; 
	}
}
