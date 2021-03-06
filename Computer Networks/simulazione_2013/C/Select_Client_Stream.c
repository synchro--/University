/* NOME:Ali Alessio  COGNOME:Salman  MATRICOLA:0000631617*/

/*Select_Client_Stream.c*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h> /*Solitamente bisogna manipolare file */ 
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <string.h> 

#define MAX_LENGTH 50

int main(int argc, char *argv[])
{
  int sd, nread;
  struct hostent *host;
  struct sockaddr_in servaddr;
  /*dichiarazioni variabili necessarie alla logica del programma*/
  char req[MAX_LENGTH], nomeFile[MAX_LENGTH], c,control; 
  int fd, i, dim; 


  /* CONTROLLO ARGOMENTI ---------------------------------- */
  if(argc!=3)
  {
    printf("Error:%s serverAddress serverPort\n", argv[0]);
    exit(1);
  }

  nread = 0;
	while (argv[2][nread] != '\0')
	{
		if ((argv[2][nread] < '0') || (argv[2][nread] > '9'))
		{
			printf("Secondo argomento non intero\n");
			exit(2);
		}
		nread++;
	}

  /* INIZIALIZZAZIONE INDIRIZZO SERVER -------------------------- */
  memset((char *)&servaddr, 0, sizeof(struct sockaddr_in));
  servaddr.sin_family = AF_INET;
  host = gethostbyname(argv[1]);
  if (host == NULL)
  {
    printf("%s not found in /etc/hosts\n", argv[1]);
    exit(1);
  }

  servaddr.sin_addr.s_addr=((struct in_addr*) (host->h_addr))->s_addr;
  servaddr.sin_port = htons(atoi(argv[2]));
  
  if (servaddr.sin_port < 1024 || servaddr.sin_port > 65535)
	{
		printf("Port scorretta...");
		exit(2);
	}



  /* CORPO DEL CLIENT:
     ciclo di accettazione di richieste da utente ------- */
  puts("inserisci tipo prenotazione da cui vuoi scaricare le immagini"); 
   
  while (gets(req))
  {
    //INTERAZIONE UTENTE

    /* CREAZIONE SOCKET ------------------------------------ */
    sd=socket(AF_INET, SOCK_STREAM, 0);
    if(sd<0) {perror("apertura socket"); exit(1);}
    printf("Client: creata la socket sd=%d\n", sd);

    /* Operazione di BIND implicita nella connect */
    if(connect(sd,(struct sockaddr *) &servaddr, sizeof(struct sockaddr))<0)
    { perror("connect"); exit(1);}
    printf("Client: connect ok\n");

    /* ----------------------------------------------------- */

    printf("tipo scelto: %s\n", req); 
          
    //invio richiesta al server
    write(sd,req,(strlen(req)+1)); //strlen(req)+1 è la stessa cosa (?) 

    /* Il client non deve piu' inviare nulla, puo' chiudere*/
    shutdown(sd,1);
 
     //leggo risposta fino ad EOF inviato dalla close del server 
      while(nread = read(sd, &control, 1) > 0 || (control!='a'))
     {
      
      printf(" ricevuto segnale %c\n", control); 
      //leggo nome file e lo apro/creo
      if(read(sd, &nomeFile, MAX_LENGTH) <0)
      {
        perror("read nome file"); 
       }
       
       if(strcmp(nomeFile, "e") == 0) {printf("Il file non esiste\n"); continue; }
       if((fd = open(nomeFile, O_CREAT | O_WRONLY | O_TRUNC, 0775)) < 0) { perror("open file"); exit(-1); }
       
       //leggo dimensione file 
       if(read(sd, &dim, sizeof(int)) < 0) perror("read dim"); 
       dim = ntohl(dim); 
       printf(" la dimensione del file : %d\n", dim); 
       
       //ricevo il file
       for(i=0; i < dim; i++)
       {
          if(read(sd, &c, 1) < 0) perror("read ricezione"); 
          if(write(fd, &c, 1) < 0) perror("write file"); 
          //write(1, &c,1); //controllo 
       }
       
       puts("Trasferimento terminato con successo "); 
       
        puts("inserisci tipo prenotazione da cui vuoi scaricare le immagini"); 
        close(fd); 
       
   }
      
    // Chiusura socket
    close(sd);

    //interazione con l'utente
  }//while

  printf("\nClient: termino...\n");
  exit(0);

} //main
