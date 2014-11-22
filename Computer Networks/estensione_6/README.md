This is a client-server application based on Java Remote Method Invocation (RMI) and socket both. 
First, the client asks the server the list of remote directories and right after, the server asks the client to choose the directory to download. Then the client send the name of the chosen directory to the server and the server returns the list of files names and files lengths and his endpoint. All of these is performed using RMI.
Afterthat the C/S communication goes on with stream sockets and the files are sent from server to client. 

Fileinfo contains the name and length of 1 file.  
RemoteInfo contains the endpoint useful for socket stream and an array of FileInfo. 




