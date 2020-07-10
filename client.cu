///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "ex3.h"

#include <random>

#define SQR(a) ((a) * (a))

long long int distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    long long int distance_sqr = 0;
    for (size_t i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        if (img_arr1[i] != img_arr2[i])
            dbg_printf("cpu[0x%4lx/0x%04lx] == 0x%x != 0x%x\n", i / (IMG_WIDTH * IMG_HEIGHT), i % (IMG_WIDTH * IMG_HEIGHT), img_arr1[i], img_arr2[i]);
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}

int randomize_images(uchar *images)
{
    std::default_random_engine generator;
    std::uniform_int_distribution<uint64_t> distribution(0,0xffffffffffffffffULL);
    for (uint64_t *p = (uint64_t *)images; p < (uint64_t *)(images + N_IMAGES * IMG_WIDTH * IMG_HEIGHT); ++p)
	*p = distribution(generator);
    return 0;
} 

int process_images(mode_enum mode, std::unique_ptr<rdma_client_context>& client)
{
    std::unique_ptr<uchar[]> images_in = std::make_unique<uchar[]>(IMG_SZ * N_IMAGES);
    std::unique_ptr<uchar[]> images_out_cpu = std::make_unique<uchar[]>(IMG_SZ * N_IMAGES);
    std::unique_ptr<uchar[]> images_out_gpu = std::make_unique<uchar[]>(IMG_SZ * N_IMAGES);

    client->set_input_images(images_in.get(), IMG_SZ * N_IMAGES);
    client->set_output_images(images_out_gpu.get(), IMG_SZ * N_IMAGES);

    double t_start, t_finish;

    /* instead of loading real images, we'll load the arrays with random data */
    printf("\n=== Randomizing images ===\n");
    t_start = get_time_msec();
    if (randomize_images(images_in.get()))
	return 1;
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    // CPU computation. For reference. Do not change
    printf("\n=== CPU ===\n");
    t_start = get_time_msec();
    for (size_t i = 0; i < N_IMAGES; i++) {
        uchar *img_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar *img_out = &images_out_cpu[i * IMG_WIDTH * IMG_HEIGHT];
        cpu_process(img_in, img_out, IMG_WIDTH, IMG_HEIGHT);
    }
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    printf("\n=== Client-Server ===\n");
    printf("mode = %s\n", mode == MODE_RPC_SERVER ? "rpc" : "queue");

    long long int distance_sqr;
    std::vector<double> req_t_start(N_IMAGES, NAN), req_t_end(N_IMAGES, NAN);

    t_start = get_time_msec();
    size_t next_img_id = 0;
    size_t num_dequeued = 0;
    const size_t total_requests = N_IMAGES * 3;

    while (next_img_id < total_requests || num_dequeued < total_requests) {
        int dequeued_img_id;
        if (client->dequeue(&dequeued_img_id)) {
            ++num_dequeued;
            req_t_end[dequeued_img_id % N_IMAGES] = get_time_msec();
        }

        /* If we are done with enqueuing, just loop until all are dequeued */
        if (next_img_id == total_requests)
            continue;

        /* Enqueue a new image */
        req_t_start[next_img_id % N_IMAGES] = get_time_msec();
        if (client->enqueue(next_img_id, &images_in[(next_img_id % N_IMAGES) * IMG_WIDTH * IMG_HEIGHT],
                                         &images_out_gpu[(next_img_id % N_IMAGES) * IMG_WIDTH * IMG_HEIGHT])) {
            ++next_img_id;
        }
    }
    t_finish = get_time_msec();
    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu.get(), images_out_gpu.get());
    printf("distance from baseline %lld (should be zero)\n", distance_sqr);
    printf("throughput = %.1lf (req/sec)\n", total_requests / (t_finish - t_start) * 1e+3);

    print_latency("overall", req_t_start, req_t_end);

    return 0;
}

int main(int argc, char *argv[]) {
    enum mode_enum mode;
    uint16_t tcp_port;

    parse_arguments(argc, argv, &mode, &tcp_port);
    if (!tcp_port) {
        printf("usage: %s <rpc|queue> <tcp port>\n", argv[0]);
        exit(1);
    }

    auto client = create_client(mode, tcp_port);

    process_images(mode, client);

    return 0;
}
