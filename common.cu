#include "ex3.h"

#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <infiniband/verbs.h>

#include <algorithm>
#include <cassert>

void parse_arguments(int argc, char **argv, enum mode_enum *mode, uint16_t *tcp_port)
{
    if (argc < 2) {
        printf("usage: %s <rpc|queue> [tcp port]\n", argv[0]);
        exit(1);
    }

    if (strcmp(argv[1], "rpc") == 0) {
        *mode = MODE_RPC_SERVER;
    } else if (strcmp(argv[1], "queue") == 0) {
        *mode = MODE_QUEUE;
    } else {
        printf("Unknown mode '%s'\n", argv[1]);
        exit(1);
    }

    if (argc < 3) {
        *tcp_port = 0;
    } else {
        *tcp_port = atoi(argv[2]);
    }
}

rdma_context::rdma_context(uint16_t tcp_port) :
    tcp_port(tcp_port)
{
}

rdma_context::~rdma_context()
{
    /* cleanup */
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(mr_requests);
    ibv_dealloc_pd(pd);
    ibv_close_device(context);

    /* we don't need TCP anymore. kill the socket */
    close(socket_fd);
}

void rdma_context::initialize_verbs(const char *device_name)
{
    /* get device list */
    struct ibv_device **device_list = ibv_get_device_list(NULL);
    if (!device_list) {
        perror("ibv_get_device_list failed");
        exit(1);
    }

    /* select device to work with */
    struct ibv_device *requested_dev = nullptr;
    for (int i = 0; device_list[i]; ++i)
        if (strcmp(device_list[i]->name, device_name)) {
            requested_dev = device_list[i];
            break;
        }
    if (!requested_dev)
        requested_dev = device_list[0];
    if (!requested_dev) {
        printf("Unable to find RDMA device '%s'\n", device_name);
        exit(1);
    }

    context = ibv_open_device(requested_dev);

    ibv_free_device_list(device_list);

    /* create protection domain (PD) */
    pd = ibv_alloc_pd(context);
    if (!pd) {
        perror("ibv_alloc_pd() failed");
        exit(1);
    }

    /* allocate a memory region for the RPC requests. */
    mr_requests = ibv_reg_mr(pd, requests.begin(), sizeof(rpc_request) * OUTSTANDING_REQUESTS, IBV_ACCESS_LOCAL_WRITE);
    if (!mr_requests) {
        perror("ibv_reg_mr() failed for requests");
        exit(1);
    }

    /* create completion queue (CQ). We'll use same CQ for both send and receive parts of the QP */
    cq = ibv_create_cq(context, 2 * OUTSTANDING_REQUESTS, NULL, NULL, 0); /* create a CQ with place for two completions per request */
    if (!cq) {
        perror("ibv_create_cq() failed");
        exit(1);
    }

    /* create QP */
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
    qp_init_attr.send_cq = cq;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.qp_type = IBV_QPT_RC; /* we'll use RC transport service, which supports RDMA */
    qp_init_attr.cap.max_send_wr = OUTSTANDING_REQUESTS; /* max of 1 WQE in-flight in SQ per request. that's enough for us */
    qp_init_attr.cap.max_recv_wr = OUTSTANDING_REQUESTS; /* max of 1 WQE in-flight in RQ per request. that's enough for us */
    qp_init_attr.cap.max_send_sge = 1; /* 1 SGE in each send WQE */
    qp_init_attr.cap.max_recv_sge = 1; /* 1 SGE in each recv WQE */
    qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) {
        perror("ibv_create_qp() failed");
        exit(1);
    }
}

void rdma_context::send_over_socket(void *buffer, size_t len)
{
    int ret = send(socket_fd, buffer, len, 0);
    if (ret < 0) {
        perror("send");
        exit(1);
    }
}

void rdma_context::recv_over_socket(void *buffer, size_t len)
{
    int ret = recv(socket_fd, buffer, len, 0);
    if (ret < 0) {
        perror("recv");
        exit(1);
    }
}

void rdma_context::send_connection_establishment_data()
{
    /* ok, before we continue we need to get info about the client' QP, and send it info about ours.
     * namely: QP number, and LID/GID.
     * we'll use the TCP socket for that */

    struct connection_establishment_data my_info = {};
    int ret;

#ifdef USE_ROCE
    /* For RoCE, GID (IP address) must by used */
    ret = ibv_query_gid(context, IB_PORT, GID_INDEX, &my_info.gid);
    if (ret) {
        perror("ibv_query_gid() failed");
        exit(1);
    }
#else
    /* For InfiniBand, LID is enough. Query port for its LID (L2 address) */
    struct ibv_port_attr port_attr;
    ret = ibv_query_port(context, IB_PORT, &port_attr);
    if (ret) {
        perror("ibv_query_port() failed");
        exit(1);
    }

    my_info.lid = port_attr.lid;
#endif

    my_info.qpn = qp->qp_num;
    send_over_socket(&my_info, sizeof(connection_establishment_data));
    print_connection_establishment_data("local", my_info);
}

connection_establishment_data rdma_context::recv_connection_establishment_data()
{
    /* ok, before we continue we need to get info about the client' QP
     * namely: QP number, and LID/GID. */

    struct connection_establishment_data remote_info;

    recv_over_socket(&remote_info, sizeof(connection_establishment_data));

    print_connection_establishment_data("remote", remote_info);

    return remote_info;
}

void rdma_context::print_connection_establishment_data(const char *type, const connection_establishment_data& data)
{
    char address[INET6_ADDRSTRLEN];

#ifdef USE_ROCE
    inet_ntop(AF_INET6, &data.gid, address, sizeof(address));
#else
    snprintf(address, sizeof(address), "LID: 0x%04x", data.lid);
#endif

    printf("  %s address:  %s, QPN 0x%06x\n", type, address, data.qpn);
}

void rdma_context::connect_qp(const connection_establishment_data &remote_info)
{
    /* this is a multi-phase process, moving the state machine of the QP step by step
     * until we are ready */
    struct ibv_qp_attr qp_attr;

    /*QP state: RESET -> INIT */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = IB_PORT;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ; /* we'll allow client to RDMA write and read on this QP */
    int ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (ret) {
        perror("ibv_modify_qp() to INIT failed");
        exit(1);
    }

    /*QP: state: INIT -> RTR (Ready to Receive) */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
#ifdef USE_ROCE
    qp_attr.path_mtu = IBV_MTU_1024;
#else
    qp_attr.path_mtu = IBV_MTU_4096;
#endif
    qp_attr.dest_qp_num = remote_info.qpn; /* qp number of the remote side */
    qp_attr.rq_psn      = 0 ;
    qp_attr.max_dest_rd_atomic = 16; /* max in-flight RDMA reads */
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.is_global = 0; /* No Network Layer (L3) */
#ifdef USE_ROCE
    qp_attr.ah_attr.grh.dgid = remote_info.gid; /* GID (L3 address) of the remote side */
    qp_attr.ah_attr.grh.sgid_index = GID_INDEX;
    qp_attr.ah_attr.grh.hop_limit = 1;
    qp_attr.ah_attr.is_global = 1;
#else
    qp_attr.ah_attr.dlid = remote_info.lid; /* LID (L2 Address) of the remote side */
#endif
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = IB_PORT;
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret) {
        perror("ibv_modify_qp() to RTR failed");
        exit(1);
    }

    /*QP: state: RTR -> RTS (Ready to Send) */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7; // 7 means infinite
    qp_attr.rnr_retry = 7; // 7 means infinite
    qp_attr.max_rd_atomic = 16;
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret) {
        perror("ibv_modify_qp() to RTS failed");
        exit(1);
    }

    /* now let's populate the receive QP with recv WQEs */
    for (int i = 0; i < OUTSTANDING_REQUESTS; i++) {
        post_recv(i);
    }
}

void rdma_context::post_recv(int index)
{
    struct ibv_recv_wr recv_wr = {}, *bad_wr; /* this is the receive work request (the verb's representation for receive WQE) */
    ibv_sge sgl = {};

    recv_wr.wr_id = index;
    if (index >= 0) {
        sgl.addr = (uintptr_t)&requests[index];
        sgl.length = sizeof(requests[0]);
        sgl.lkey = mr_requests->lkey;
    }
    recv_wr.sg_list = &sgl;
    recv_wr.num_sge = 1;
    if (int ret = ibv_post_recv(qp, &recv_wr, &bad_wr)) {
	errno = ret;
        perror("ibv_post_recv() failed");
        exit(1);
    }
}

void rdma_context::post_rdma_read(void *local_dst, uint32_t len, uint32_t lkey, uint64_t remote_src, uint32_t rkey, uint64_t wr_id)
{
    ibv_sge sgl = {
        (uint64_t)(uintptr_t)local_dst,
        len,
        lkey
    };

    ibv_send_wr send_wr = {};
    ibv_send_wr *bad_send_wr;

    send_wr.opcode = IBV_WR_RDMA_READ;
    send_wr.wr_id = wr_id;
    send_wr.sg_list = &sgl;
    send_wr.num_sge = 1;
    send_wr.send_flags = IBV_SEND_SIGNALED;
    send_wr.wr.rdma.remote_addr = remote_src;
    send_wr.wr.rdma.rkey = rkey;

    if (ibv_post_send(qp, &send_wr, &bad_send_wr)) {
	perror("ibv_post_send() failed");
	exit(1);
    }
}

void rdma_context::post_rdma_write(uint64_t remote_dst, uint32_t len, uint32_t rkey,
		     void *local_src, uint32_t lkey, uint64_t wr_id,
		     uint32_t *immediate)
{
    ibv_sge sgl = {
        (uint64_t)(uintptr_t)local_src,
        len,
        lkey
    };

    ibv_send_wr send_wr = {};
    ibv_send_wr *bad_send_wr;

    if (immediate) {
        send_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        send_wr.imm_data = *immediate;
    } else {
        send_wr.opcode = IBV_WR_RDMA_WRITE;
    }
    send_wr.wr_id = wr_id;
    send_wr.sg_list = &sgl;
    send_wr.num_sge = 1;
    send_wr.send_flags = IBV_SEND_SIGNALED;
    send_wr.wr.rdma.remote_addr = remote_dst;
    send_wr.wr.rdma.rkey = rkey;

    if (ibv_post_send(qp, &send_wr, &bad_send_wr)) {
	perror("ibv_post_send() failed");
	exit(1);
    }
}

rdma_server_context::rdma_server_context(uint16_t tcp_port) :
    rdma_context(tcp_port)
{
    /* Initialize memory and CUDA resources */
    CUDA_CHECK(cudaMallocHost(&images_in, OUTSTANDING_REQUESTS * IMG_SZ));
    CUDA_CHECK(cudaMallocHost(&images_out, OUTSTANDING_REQUESTS * IMG_SZ));

    /* Create a TCP connection to exchange InfiniBand parameters */
    tcp_connection();

    /* Open up some InfiniBand resources */
    initialize_verbs(IB_DEVICE_NAME_SERVER);

    /* register a memory region for the input / output images. */
    mr_images_in = ibv_reg_mr(pd, images_in, OUTSTANDING_REQUESTS * IMG_SZ, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr_images_in) {
        perror("ibv_reg_mr() failed for input images");
        exit(1);
    }

    /* register a memory region for the input / output images. */
    mr_images_out = ibv_reg_mr(pd, images_out, OUTSTANDING_REQUESTS * IMG_SZ, IBV_ACCESS_REMOTE_READ);
    if (!mr_images_out) {
        perror("ibv_reg_mr() failed for output images");
        exit(1);
    }

    /* Receive RDMA parameters with the client */
    connection_establishment_data client_info = recv_connection_establishment_data();

    /* now need to connect the QP to the client's QP. */
    connect_qp(client_info);

    /* Send RDMA parameters to the client */
    send_connection_establishment_data();
}

rdma_server_context::~rdma_server_context()
{
    ibv_dereg_mr(mr_images_in);
    ibv_dereg_mr(mr_images_out);

    CUDA_CHECK(cudaFreeHost(images_in));
    CUDA_CHECK(cudaFreeHost(images_out));

    close(listen_fd);
}

void rdma_server_context::tcp_connection()
{
    /* setup a TCP connection for initial negotiation with client */
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        perror("socket");
        exit(1);
    }
    listen_fd = lfd;

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(struct sockaddr_in));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(tcp_port);

    int one = 1;
    if (setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one))) {
        perror("SO_REUSEADDR");
        exit(1);
    }

    if (bind(lfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
        perror("bind");
        exit(1);
    }

    if (listen(lfd, 1)) {
        perror("listen");
        exit(1);
    }

    printf("Server waiting on port %d. Client can connect\n", tcp_port);

    int sfd = accept(lfd, NULL, NULL);
    if (sfd < 0) {
        perror("accept");
        exit(1);
    }
    printf("client connected\n");
    socket_fd = sfd;
}

rdma_client_context::rdma_client_context(uint16_t tcp_port) :
    rdma_context(tcp_port)
{
    /* Create a TCP connection to exchange InfiniBand parameters */
    tcp_connection();

    /* Open up some InfiniBand resources */
    initialize_verbs(IB_DEVICE_NAME_CLIENT);

    /* exchange InfiniBand parameters with the client */
    send_connection_establishment_data();
    connection_establishment_data server_info = recv_connection_establishment_data();

    /* now need to connect the QP to the client's QP. */
    connect_qp(server_info);
}

rdma_client_context::~rdma_client_context()
{
}

void rdma_client_context::tcp_connection()
{
    /* first we'll connect to server via a TCP socket to exchange InfiniBand parameters */
    int sfd;
    sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) {
        perror("socket");
        exit(1);
    }

    struct sockaddr_in server_addr;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); /* server is on same machine */
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(tcp_port);

    if (connect(sfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
        perror("connect");
        exit(1);
    }

    printf("connected\n");
    socket_fd = sfd;
}

void print_latency(const char *type, const std::vector<double>& req_t_start, const std::vector<double>& req_t_end)
{
    std::vector<double> latencies(req_t_start.size());
    double avg_latency = 0;
    for (size_t i = 0; i < latencies.size(); i++) {
        double cur_latency = req_t_end[i] - req_t_start[i];
        if (isnan(req_t_start[i]) || isnan(req_t_end[i])) {
            printf("Missing measurement for req %ld: start=%lf end=%lf\n", i, req_t_start[i], req_t_end[i]);
            exit(1);
        }
	latencies[i] = cur_latency;
        avg_latency += cur_latency;
    }
    avg_latency /= latencies.size();
    std::sort(latencies.begin(), latencies.end());

    printf("%8s latency [msec]:\n%12s%12s%12s%12s%12s\n", type, "avg", "min", "median", "99th perc.", "max");
    printf("%12.4lf%12.4lf%12.4lf%12.4lf%12.4lf\n", avg_latency, latencies[0], latencies[latencies.size() / 2], latencies[latencies.size() * 99 / 100], latencies[latencies.size() - 1]);
}
