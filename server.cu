///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <string.h>
#include <assert.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "ex3.h"

#define TCP_PORT_OFFSET 23456
#define TCP_PORT_RANGE 1000

int main(int argc, char *argv[]) {
    std::unique_ptr<rdma_server_context> server;

    enum mode_enum mode;
    uint16_t tcp_port;

    parse_arguments(argc, argv, &mode, &tcp_port);
    if (!tcp_port) {
        srand(time(NULL));
        tcp_port = TCP_PORT_OFFSET + (rand() % TCP_PORT_RANGE); /* to avoid conflicts with other users of the machine */
    }

    server = create_server(mode, tcp_port);
    if (!server) {
        printf("Error creating server context.\n");
        exit(1);
    }

    /* now finally we get to the actual work, in the event loop */
    /* The event loop can be used for queue mode for the termination message */
    server->event_loop();

    printf("Done\n");

    return 0;
}
