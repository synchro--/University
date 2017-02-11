### Assignment: C/S File transfer application based on Java RMI + Java Socket

This is a client-server file transfer application based on Java Remote Method Invocation (RMI) and java socket both. 
First, the client asks the server the list of remote directories and right after, the server asks the client to choose the directory to download. Then the client send the name of the chosen directory to the server and the server returns the list of files names and files lengths and its endpoint. All of former is performed using RMI.
After that the C/S communication goes on with *stream sockets* and the files are sent from server to client. 

- Fileinfo contains the name and length of 1 file.  
- RemoteInfo contains the endpoint useful for socket stream and an array of FileInfo.

*N.B.: this code sucks and it's old. One day, when dragon Shenron will be up in the dark sky, I'll find the time to fix it, hopefully.*




