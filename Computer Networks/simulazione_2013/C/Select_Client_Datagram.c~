/* NOME:Ali Alessio  COGNOME:Salman  MATRICOLA:0000631617*/

/*Select_Client_Datagram*/ 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#define MAX_LENGTH 50
int main(int argc, char **argv)
{
 struct hostent *host;
 struct sockaddr_in clientaddr, servaddr;
 int port, nread, sd, len = 0;
 /*dichiarazione variabili necessarie alla logica del programma*/
 char id[MAX_LENGTH],c; 
 int persone,esito; 
 

  /* CONTROLLO ARGOMENTI ---------------------------------- */
  if(argc!=3)
  {
    printf("Error:%s serverAddress serverPort\n", argv[0]);
    exit(1);
  }

  /* INIZIALIZZAZIONE INDIRIZZO CLIENT E SERVER --------------------- */
  memset((char *)&clientaddr, 0, sizeof(struct sockaddr_in));
  clientaddr.sin_family = AF_INET;
  host=gethostbyname("localhost");
  clientaddr.sin_addr.s_addr=((struct in_addr *)(host->h_addr))->s_addr;
  if( clientaddr.sin_addr.s_addr == INADDR_NONE )
  {
    perror("Bad address");
    exit(2);
  }

  clientaddr.sin_port = 0;

  memset((char *)&servaddr, 0, sizeof(struct sockaddr_in));
  servaddr.sin_family = AF_INET;
  host = gethostbyname (argv[1]);

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

	port = atoi(argv[2]);

	if (port < 1024 || port > 65535)
	{
		printf("Port scorretta...");
		exit(2);
	}
	
  if (host == NULL)
  {
    printf("%s not found in /etc/hosts\n", argv[1]);
    exit(2);
  }
  else
  {
    servaddr.sin_addr.s_addr=((struct in_addr *)(host->h_addr))->s_addr;
    servaddr.sin_port = htons(port);
  }


  /* CREAZIONE SOCKET ---------------------------------- */
  sd=socket(AF_INET, SOCK_DGRAM, 0);
  if(sd<0) {perror("apertura socket"); exit(1);}
  printf("Client: creata la socket sd=%d\n", sd);

  /* BIND SOCKET*/
  if(bind(sd,(struct sockaddr *) &clientaddr, sizeof(clientaddr))<0)
  {perror("bind socket "); exit(1);}
  printf("Client: bind socket ok, alla porta %i\n", clientaddr.sin_port);

  /* CORPO DEL CLIENT:
     ciclo di accettazione di richieste da utente ------- */
     puts("inserisci id"); 

  while (gets(id))
  { 
    printf("id scelto : %s ", id); 
    /* richiesta operazione */
    if(sendto(sd, id, sizeof(id), 0, (struct sockaddr *)&servaddr, sizeof(servaddr))<0)
    {
      perror("sendto 1");
    }

    puts("inserisci numero");  
        while(scanf("%d",&persone) != 1)
        {
           do  c = getchar(); 
           while(getchar()!='\n'); 
            puts("inserisci numero");  
        }
        gets(id); //consumo il fine linea 
        
        persone = htonl(persone); 
        
         if(sendto(sd, &persone, sizeof(int), 0, (struct sockaddr *)&servaddr, sizeof(servaddr))<0)
    {
      perror("sendto 2");
    }
        
    /* ricezione del risultato */
    printf("In attesa di risposta...\n");

    len=sizeof(servaddr);
    if (recvfrom(sd, &esito,sizeof(int), 0, (struct sockaddr *)&servaddr, &len)<0)
    {perror("recvfrom");}
    
    esito = ntohl(esito); 
    printf("esito : %d\n",esito);
    if(esito == -1) printf("l'id scelto non esiste\n"); 
    if(esito == 0) puts("operazione eseguita con successo"); 

    //interazione utente
    puts("inserisci id"); 

  } // while gets


  close(sd);
  printf("\nClient: termino...\n");
  exit(0);
}
