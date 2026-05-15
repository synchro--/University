/* NOME:Ali Alessio  COGNOME:Salman  MATRICOLA:0000631617*/

/* Select_Server.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#define N 5
#define MAX_LENGTH 50
#define max(a,b) ((a) > (b) ? (a) : (b))

typedef struct {

char id[MAX_LENGTH]; 
int  persone; 
char prenotazione[MAX_LENGTH]; 
char veicolo[MAX_LENGTH]; 
char targa[MAX_LENGTH];
char img_file[MAX_LENGTH]; 
} Prenotazione; 


/*eventuali funzioni utili per la logica del programma*/ 


/********************************************************/

void gestore(int signo)
{
	int stato;
	printf("esecuzione gestore di SIGCHLD\n");
	wait(&stato);
}

/********************************************************/

int main(int argc, char **argv)
{
	int nread,listenfd, connfd, udpfd, nready, maxfdp1;
	const int on = 1;

	fd_set rset;
	struct hostent *host;
	struct hostent *clienthost;
	int len, port;
	struct sockaddr_in clientaddr, servaddr;
	
	Prenotazione db[N]; //database
	char id[MAX_LENGTH] , tipo[MAX_LENGTH], c;
	int persone = -1, dim, fd , i; 
	int esito , numFiles;
	int trovato = -1; 

	/* CONTROLLO ARGOMENTI ---------------------------------- */
	if (argc != 2)
	{
		printf("Error: %s port\n", argv[0]);
		exit(1);
	}
	
	nread = 0;
	while (argv[1][nread] != '\0')
	{
		if ((argv[1][nread] < '0') || (argv[1][nread] > '9'))
		{
			printf(" argomento non intero\n");
			exit(2);
		}
		nread++;
	}
	
	 port = atoi(argv[1]);
	
	if (port < 1024 || port > 65535)
	{
		printf("Port scorretta...");
		exit(2);
	}

	printf("Server avviato\n");
	
	//INIZIALIZZAZIONE
	
        for(i=0; i < N; i++)
        {
            strcpy(db[i].id,"L"); strcpy(db[i].veicolo,"L"); strcpy(db[i].prenotazione,"L");
            strcpy(db[i].targa,"L"); strcpy(db[i].img_file,"L"); db[i].persone = -1; 
        }
        
        strcpy(db[0].prenotazione,"piazzola"); strcpy(db[1].prenotazione,"piazzola"); 
        strcpy(db[0].img_file,"ciao.txt"); strcpy(db[1].img_file,"bridge.jpeg"); 
        strcpy(db[0].id,"id"); 

	/* CREAZIONE SOCKET TCP ------------------------------------------------------ */
	listenfd = socket(AF_INET, SOCK_STREAM, 0);
	if (listenfd < 0)
	{
		perror("apertura socket TCP ");
		exit(1);
	}
	printf("Creata la socket TCP d'ascolto, fd=%d\n", listenfd);

	/* INIZIALIZZAZIONE INDIRIZZO SERVER ----------------------------------------- */
	memset((char *) &servaddr, 0, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = INADDR_ANY;
	servaddr.sin_port = htons(port);

	if (setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0)
	{
		perror("set opzioni socket TCP");
		exit(2);
	}
	printf("Set opzioni socket TCP ok\n");

	if (bind(listenfd, (struct sockaddr *) &servaddr, sizeof(servaddr)) < 0)
	{
		perror("bind socket TCP");
		exit(3);
	}
	printf("Bind socket TCP ok\n");

	if (listen(listenfd, 5) < 0)
	{
		perror("listen");
		exit(4);
	}
	printf("Listen ok\n");

	/* CREAZIONE SOCKET UDP ----------------------------------------------------- */
	udpfd = socket(AF_INET, SOCK_DGRAM, 0);
	if (udpfd < 0)
	{
		perror("apertura socket UDP");
		exit(5);
	}
	printf("Creata la socket UDP, fd=%d\n", udpfd);

	/* INIZIALIZZAZIONE INDIRIZZO SERVER ---------------------------------------- */
	memset((char *) &servaddr, 0, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = INADDR_ANY;
	servaddr.sin_port = htons(port);

	if (setsockopt(udpfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0)
	{
		perror("set opzioni socket UDP");
		exit(6);
	}
	printf("Set opzioni socket UDP ok\n");

	if (bind(udpfd, (struct sockaddr *) &servaddr, sizeof(servaddr)) < 0)
	{
		perror("bind socket UDP");
		exit(7);
	}
	printf("Bind socket UDP ok\n");

	/* AGGANCIO GESTORE PER EVITARE FIGLI ZOMBIE -------------------------------- */
	signal(SIGCHLD, gestore);

	/* PULIZIA E SETTAGGIO MASCHERA DEI FILE DESCRIPTOR ------------------------- */
	FD_ZERO(&rset);
	maxfdp1 = max(listenfd, udpfd) + 1;

	/* CICLO DI RICEZIONE EVENTI DALLA SELECT ----------------------------------- */
	for (;;)
	{
		FD_SET(listenfd, &rset);
		FD_SET(udpfd, &rset);

		if ((nready = select(maxfdp1, &rset, NULL, NULL, NULL)) < 0)
		{
			if (errno == EINTR) continue;
			else
			{
				perror("select");
				exit(8);
			}
		}

		/* GESTIONE RICHIESTE DI STREAM CON GENERAZIONE DI FIGLIO PER OGNI CICLO */

		if (FD_ISSET(listenfd, &rset))
		{
			printf("Ricevuta richiesta...\n"); //da modificare

			len = sizeof(struct sockaddr_in);
			if ((connfd = accept(listenfd, (struct sockaddr *) &clientaddr, &len))
					< 0)
			{
				if (errno == EINTR) continue;
				else
				{
					perror("accept");
					exit(9);
				}
			}

			if (fork() == 0)
			{ /* processo figlio che serve l'operazione richiesta */
				close(listenfd);
				host = gethostbyaddr((char *) &clientaddr.sin_addr,
						sizeof(clientaddr.sin_addr), AF_INET);
				if (host == NULL)
				{
					printf("client host information not found\n");
				}
				else printf("Server (figlio): host client e' %s \n", host->h_name);

                                 // Leggo la richiesta del client ed eseguo 
                                 if(read(connfd, &tipo, MAX_LENGTH) > 0)
                                 {
                                 printf("ricevuto tipo %s\n", tipo); 
                                 trovato = -1;
                                   for(i=0; i < N; i++)
                                   {
                                     if(strcmp(db[i].prenotazione, tipo) ==0)
                                     {
                                       trovato = 0; 
                                       /* invio un messaggio di inizio interazione */
                                       if(write(connfd, "a", 1) <0) perror("Write inizio"); 
                                        //apro il file
                                       printf("apro il file %s\n", db[i].img_file); 
                                       if( (fd = open(db[i].img_file,O_RDONLY)) < 0)
                                       {
                                         perror("open file"); write(connfd, "e", 1); //invio codice errore
                                         continue; 
                                        }
                                       //invio nome immagine 
                                       if(write(connfd, db[i].img_file, (strlen(db[i].img_file) +1)) < 0)
                                       {
                                          perror("write nome file"); 
                                          exit(-2); 
                                        
                                        }
                                        
                                        dim =0; 
                                        //conto dimensione e invio 
                                        while((nread = read(fd, &c, 1)) >0)
                                        {
                                           dim++; 
                                           //printf(" letto %c\n", c);
                                         }
                                         
                                         printf("dimensione file: %d\n", dim); 
                                         dim = htonl(dim); 
                                         printf("dim da inviare : %d\n", dim); 
                                         
                                         if(write(connfd, &dim, sizeof(int)) < 0)
                                         { perror("write dim"); exit(-3); }
                                         dim = ntohl(dim); 
                                         
                                         //invio il file 
                                         printf("invio il file....\n"); 
                                         
                                         
                                         lseek(fd, 0,SEEK_SET); 
                                         while((nread = read(fd, &c, 1)) > 0)
                                         {
                                            if((write(connfd, &c, 1)) < 0) perror("write bytes "); 
                                            //write(1, &c,1); //controllo 
                                          }
                                     }
                                     
                                     
                                  }   
                                  
                                  if(trovato == -1) write(connfd, "e",1); 
                                  
                                  //chiudo la connessione
                                  close(fd);
                                  close(connfd); //invio EOF al pari
                                  printf("figlio: termino .."); 
                                  exit(0); //IL FIGLIO TERMINA 
                                  
                                  /*Se il figlio non terminasse la select proverebbe al
                                  al prossimo ciclo ad aprire una comunicazione su una socket (connfd) 
                                  ancora occupata e quindi darebbe errore terminando il figlio. 
                                  Seguito dal gestore del SIGCHILD */ 
  
                                 }
  
                                     else perror("read: ");

			}//figlio

		/* padre chiude la socket dell'operazione */
	        close(connfd);

		} //gestione richieste stream
		
		
		

		/* GESTIONE RICHIESTE DATAGRAM */
		if (FD_ISSET(udpfd, &rset))
		{
			len = sizeof(struct sockaddr_in);
			//ricevo la richiesta dal client
			if (recvfrom(udpfd, &id, sizeof(id), 0,
					(struct sockaddr *) &clientaddr, &len) < 0)
			{
				perror("recvfrom ");
				continue;
			}
			
			len = sizeof(struct sockaddr_in);
			//ricevo la richiesta dal client
			if (recvfrom(udpfd, &persone, sizeof(int), 0,
					(struct sockaddr *) &clientaddr, &len) < 0)
			{
				perror("recvfrom ");
				continue;
		        }
			clienthost = gethostbyaddr((char *) &clientaddr.sin_addr,
					sizeof(clientaddr.sin_addr), AF_INET);
			if (clienthost == NULL) printf("client host information not found\n");
			else printf("Operazione richiesta da: %s %i\n", clienthost->h_name,
					(unsigned) ntohs(clientaddr.sin_port));
          
                       // printf( "dada %d", persone); 
                        persone = ntohl(persone); 
                        printf("aggiornamento della prenotazione con id:%s di persone: %d\n", id,persone); 
		        //QUI LOGICA DEL SERVIZIO UDP
		        esito = -1; 
		        for(i=0; i < N; i++)
		        {
		          if(strcmp(db[i].id, id) == 0)
		          {
		             esito = 0; 
		             db[i].persone = persone; 
		          }
		         }

                         esito = htonl(esito); 
                        //Invio il risultato al Client (res è la variabile contenente il risultato)
			if (sendto(udpfd, &esito, sizeof(int), 0, (struct sockaddr *) &clientaddr,
					len) < 0)
			{
				perror("sendto ");
				continue;
			}

		}  /*fine gestione richieste DATAGRAM*/ 


	} /* ciclo for della select */

	/* NEVER ARRIVES HERE */
	exit(0);

}//main

