///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <memory>
#include <vector>

#include <infiniband/verbs.h>

#define IMG_HEIGHT 64
#define IMG_WIDTH 64
#define IMG_SZ (IMG_WIDTH * IMG_HEIGHT)

#define N_COLORS 4

typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define VERBS_WC_CHECK(wc) do { \
    if ((wc).status != IBV_WC_SUCCESS) { \
        printf("ERROR: got CQE with error '%s' (%d) (%s:%d)\n", ibv_wc_status_str(wc.status), wc.status, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

#ifndef DEBUG
#define dbg_printf(...)
#else
#define dbg_printf(...) do { printf(__VA_ARGS__); } while (0)
#endif

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}

void cpu_process(uchar *img_in, uchar *img_out, int width, int height);

struct rpc_request
{
    int request_id; /* Returned to the client via RDMA write immediate value; use -1 to terminate */

    /* Input buffer */
    int input_rkey;
    int input_length;
    uint64_t input_addr;

    /* Output buffer */
    int output_rkey;
    int output_length;
    uint64_t output_addr;
};

#define IB_DEVICE_NAME_SERVER ""
#define IB_DEVICE_NAME_CLIENT ""
#define IB_PORT 1
#define GID_INDEX 3

#define OUTSTANDING_REQUESTS (8*1024)

enum mode_enum {
    MODE_RPC_SERVER,
    MODE_QUEUE,
};

void parse_arguments(int argc, char **argv, enum mode_enum *mode, uint16_t *tcp_port);

#define USE_ROCE

/* Data to exchange between client and server for communication */
struct connection_establishment_data {
#ifdef USE_ROCE
    ibv_gid gid;
#else
    int lid;
#endif
    int qpn;
};

/* Helper functions for establishing an RDMA connection, both on the server and
 * on the client */
class rdma_context
{
protected:
    uint16_t tcp_port;
    int socket_fd; /* Connected socket for TCP connection */

    /* InfiniBand/verbs resources */
    struct ibv_context *context = nullptr;
    struct ibv_pd *pd = nullptr;
    struct ibv_qp *qp = nullptr;
    struct ibv_cq *cq = nullptr;

    std::array<rpc_request, OUTSTANDING_REQUESTS> requests; /* Array of outstanding requests received from the network */
    struct ibv_mr *mr_requests = nullptr; /* Memory region for RPC requests */

    void initialize_verbs(const char *device_name);
    void send_over_socket(void *buffer, size_t len);
    void recv_over_socket(void *buffer, size_t len);
    void send_connection_establishment_data();
    connection_establishment_data recv_connection_establishment_data();
    static void print_connection_establishment_data(const char *type, const connection_establishment_data& data);
    void connect_qp(const connection_establishment_data& remote_info);

    /* Post a receive buffer of the given index (from the requests array) to the receive queue */
    void post_recv(int index = -1);

    /* Helper function to post an asynchronous RDMA Read request */
    void post_rdma_read(void *local_dst, uint32_t len, uint32_t lkey,
                        uint64_t remote_src, uint32_t rkey, uint64_t wr_id);
    void post_rdma_write(uint64_t remote_dst, uint32_t len, uint32_t rkey,
			 void *local_src, uint32_t lkey, uint64_t wr_id,
			 uint32_t *immediate = NULL);

public:
    explicit rdma_context(uint16_t tcp_port);
    virtual ~rdma_context();
};

/* Abstract server class for RPC and remote queue servers */
class rdma_server_context : public rdma_context
{
private:
    int listen_fd; /* Listening socket for TCP connection */

public:
    explicit rdma_server_context(uint16_t tcp_port);

    virtual ~rdma_server_context();
    virtual void event_loop() = 0;

protected:
    void tcp_connection();

    uchar *images_in; /* Input images for all outstanding requests */
    uchar *images_out; /* Output images for all outstanding requests */
    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port);

/* Abstract client class for RPC and remote queue parts of the exercise */
class rdma_client_context : public rdma_context
{
private:

public:
    explicit rdma_client_context(uint16_t tcp_port);

    virtual ~rdma_client_context();

    virtual void set_input_images(uchar *images_in, size_t bytes) = 0;
    virtual void set_output_images(uchar *images_out, size_t bytes) = 0;

    /* Enqueue an image for processing. Receives pointers to pinned host
     * memory. Return false if there is no room for image (caller will try again).
     */
    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) = 0;

    /* Checks whether an image has completed processing. If so, set img_id
     * accordingly, and return true. */
    virtual bool dequeue(int *img_id) = 0;

protected:
    void tcp_connection();
};

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port);

#define N_IMAGES 50000ULL
void print_latency(const char *type, const std::vector<double>& req_t_start, const std::vector<double>& req_t_end);

///////////////////////////////////////////////////////////////////////////////////////////////////////////

