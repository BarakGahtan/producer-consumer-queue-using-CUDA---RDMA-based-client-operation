# producer-consumer-queue-using-CUDA---RDMA-based-client-operation
 implement an RDMA-based client application performing the algorithm from homework 1 
 using the producer-consumer queue you developed in homework 2. network client-server version of the image quantization server 
 from homework 2 using RDMA Verbs. We will control the CPU-GPU queues remotely using RDMA from the client.  
 As an example, you are provided with an RPC-based implementation. 
 Your task will be to implement a second version in which the client uses RDMA to control the GPU.  
 RPC Protocol  This version is given as an example you can use to learn how to implement the second part. No need to implement this part yourselves.   
 The main RPC protocol is: 
1. The client sends a request to the server using a Send operation. Each request 
contains a unique identifier chosen by the client, and the parameters the server needs to access the input and output images remotely (Key, Address). 

2. The server performs RDMA Read operation to read the input image from the addressspecified by the client. 
3. The server performs its task (image quantization) using the GPU.
4. The server uses an RDMA Write with Immediate operation to send the output image back to the client at the requested address. The immediate value is the unique identifier chosen by the client for this request in step 1. 
back to the client at the requested address. The immediate value is the unique identifier chosen by the client for this request in step 1. 
5. The client receives a completion, notices the task has completed and continues to the
next task. 
We use a special RPC request to indicate that the server needs to terminate. Such a request causes the server to terminate without doing steps 2-3, replying with an empty RDMA Write with Immediate operation (step 4) to let the client know it is terminating. 
You can find the server code for this protocol in the server_rpc_context class and the client code in the client_rpc_context class. 

full details in the PDF. 
