#include "ex2.cu"
#include "ex3.h"
#include <cassert>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <unistd.h>
#define ISFREE -1

class server_rpc_context : public rdma_server_context {
private:
    std::unique_ptr<gpu_image_processing_context> gpu_context;

public:
    explicit server_rpc_context(uint16_t tcp_port) : rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
    }

    virtual void event_loop() override
    {
        /* so the protocol goes like this:
         * 1. we'll wait for a CQE indicating that we got an Send request from the client.
         *    this tells us we have new work to do. The wr_id we used in post_recv tells us
         *    where the request is.
         * 2. now we send an RDMA Read to the client to retrieve the request.
         *    we will get a completion indicating the read has completed.
         * 3. we process the request on the GPU.
         * 4. upon completion, we send an RDMA Write with immediate to the client with
         *    the results.
         */
        rpc_request* req;
        uchar *img_in;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];

                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        goto send_rdma_write;
                    }

                    /* Step 2: send RDMA Read to client to read the input */
                    post_rdma_read(
                        img_in,             // local_src
                        req->input_length,  // len
                        mr_images_in->lkey, // lkey
                        req->input_addr,    // remote_dst
                        req->input_rkey,    // rkey
                        wc.wr_id);          // wr_id
                    break;

                case IBV_WC_RDMA_READ:
                    /* Completed RDMA read for a request */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];
                    img_out = &images_out[wc.wr_id * IMG_SZ];

                    // Step 3: Process on GPU
                    gpu_context->enqueue(wc.wr_id, img_in, img_out);
		    break;
                    
                case IBV_WC_RDMA_WRITE:
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

		    if (terminate)
			got_last_cqe = true;

                    break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }

            // Dequeue completed GPU tasks
            int dequeued_img_id;
            if (gpu_context->dequeue(&dequeued_img_id)) {
                req = &requests[dequeued_img_id];
                img_out = &images_out[dequeued_img_id * IMG_SZ];

send_rdma_write:
                // Step 4: Send RDMA Write with immediate to client with the response
		post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_img_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate
            }
        }
    }
};

class client_rpc_context : public rdma_client_context {
private:
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
public:
    explicit client_rpc_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
    }

    ~client_rpc_context()
    {
        kill();
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        if (requests_sent - send_cqes_received == OUTSTANDING_REQUESTS)
            return false;

        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = img_id;
        req->input_rkey = img_in ? mr_images_in->rkey : 0;
        req->input_addr = (uintptr_t)img_in;
        req->input_length = IMG_SZ;
        req->output_rkey = img_out ? mr_images_out->rkey : 0;
        req->output_addr = (uintptr_t)img_out;
        req->output_length = IMG_SZ;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = img_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* When WQE is completed we expect a CQE */
        /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
        /* The order between the two is not guarenteed */

        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            ++send_cqes_received;
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *img_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }

        /* step 2: post receive buffer for the next RPC call (next RDMA write with imm) */
        post_recv();

        return true;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL)) ;
        int img_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&img_id);
        } while (!dequeued || img_id != -1);
    }
};
int calc_blocks(int threads_per_block)
{
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int maxByRegsPerSM = prop.regsPerMultiprocessor / threads_per_block / 32;
    int maxBySharedMemory = prop.sharedMemPerMultiprocessor / 1312;
    int maxByThreads = prop.maxThreadsPerMultiProcessor / threads_per_block;
    printf("maxByRegsPerSM: %d\nmaxBySharedMemoryPerSM: %d\nmaxByThreadsPerSM: %d\n", maxByRegsPerSM, maxBySharedMemory, maxByThreads);
    auto blocks = min(min(maxByRegsPerSM, maxBySharedMemory), maxByThreads) * prop.multiProcessorCount;
    printf("number of blocks: %d\n", blocks);
    return blocks;
}

typedef struct SERVER_INFO {
    uint64_t addr_images_in;
    uint32_t rkey_images_in;
    uint64_t addr_images_out;
    uint32_t rkey_images_out;
} server_info;

typedef struct CLIENT_INFO{
    uint64_t addr_images_in;
    uint32_t lkey_images_in;
    uint64_t addr_images_out;
    uint32_t lkey_images_out;
} client_info;

typedef struct Q_INFO{
    uint32_t rkey_gpu_to_cpu;
    uint64_t addr_images_in;
    uint32_t rkey_images_in;
    uint64_t addr_images_out;
    uint32_t rkey_images_out;
    queue<cpu_to_gpu_entry> *add_cpu_to_gpu;
    uint32_t rkey_cpu_to_gpu;
    queue<gpu_to_cpu_entry> *add_gpu_to_cpu;
    int blocks;
} q_info;

template <typename T> class cli_q {
public:
    int pi, ci;
    struct ibv_mr *mr_pi,*mr_ci;
    uint32_t rkey;
    queue<T>* addr;
    cli_q(queue<T> *queue_add, uint32_t given_rkey, ibv_pd *pd){
        addr = queue_add;
        rkey = given_rkey;
        mr_pi = ibv_reg_mr(pd, &pi, sizeof(int),  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_pi) exit(1);
        mr_ci = ibv_reg_mr(pd, &ci, sizeof(int ),  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_ci) exit(1);
    }
    ~cli_q(){
        ibv_dereg_mr(mr_pi);
        ibv_dereg_mr(mr_ci);
    }
};

class server_queues_context : public rdma_server_context {
private:
    queue<cpu_to_gpu_entry> *cpu2gpu;
    queue<gpu_to_cpu_entry> *gpu2cpu;
    struct ibv_mr *mr_cpu_gpu,*mr_gpu_cpu; /* Memory region for CPU-GPU queue */
    int blocks;

    void allocation(){
        CUDA_CHECK(cudaHostAlloc(&cpu2gpu, sizeof(queue<cpu_to_gpu_entry>)*blocks, 0));
        cpu2gpu = new (cpu2gpu) queue<cpu_to_gpu_entry>[blocks];
        mr_cpu_gpu = ibv_reg_mr(pd, cpu2gpu, sizeof(queue<cpu_to_gpu_entry>)*blocks, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE );//TODO: is it pd?
        if (!mr_cpu_gpu) exit(1);
        CUDA_CHECK(cudaHostAlloc(&gpu2cpu, sizeof(queue<gpu_to_cpu_entry>)*blocks, 0));
        gpu2cpu = new (gpu2cpu) queue<gpu_to_cpu_entry>[blocks];
        mr_gpu_cpu = ibv_reg_mr(pd, gpu2cpu, sizeof(queue<gpu_to_cpu_entry>)*blocks, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE );//TODO: is it pd?
        if (!mr_gpu_cpu) exit(1);
    }

    void init_info(q_info* q_info){
        q_info->add_cpu_to_gpu = cpu2gpu;
        q_info->rkey_cpu_to_gpu = mr_cpu_gpu->rkey;
        q_info->add_gpu_to_cpu = gpu2cpu;
        q_info->rkey_gpu_to_cpu = mr_gpu_cpu->rkey;
        q_info->addr_images_in = (uint64_t)images_in;
        q_info->rkey_images_in = mr_images_in->rkey;
        q_info->addr_images_out = (uint64_t)images_out;
        q_info->rkey_images_out = mr_images_out->rkey;
        q_info->blocks = blocks;
    }
public:
    explicit server_queues_context(uint16_t tcp_port) : rdma_server_context(tcp_port) {
        blocks = calc_blocks(256);
        q_info q_info;
        this->allocation();
        this->init_info(&q_info);
        send_over_socket(&q_info, sizeof(q_info));
        gpu_process_image_consumer<<<blocks, 256>>>(cpu2gpu, gpu2cpu);
    }

    ~server_queues_context() {
        for (int i = 0; i < blocks; i++){
            cpu2gpu[i].kill.store(true, memory_order_relaxed);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFreeHost(cpu2gpu));
        CUDA_CHECK(cudaFreeHost(gpu2cpu));
        ibv_dereg_mr(mr_cpu_gpu);
        ibv_dereg_mr(mr_gpu_cpu);
    }

    virtual void event_loop() override {
        bool kill = false;
        recv_over_socket(&kill, sizeof(bool));
    }
};

class client_queues_context : public rdma_client_context {
private:
    struct ibv_mr *mr_images_in,*mr_images_out; 
    struct cpu_to_gpu_entry cpu_gpu_entry;
    struct ibv_mr *mr_cpu_gpu_entry,*mr_gpu_cpu_entry;;
    struct gpu_to_cpu_entry gpu_cpu_entry;
    cli_q<cpu_to_gpu_entry> *cpu_to_gpu;
    cli_q<gpu_to_cpu_entry> *gpu_to_cpu;
    client_info cli_info;
    server_info ser_info;
    int wr_id,blocks;
    int server_space[OUTSTANDING_REQUESTS];

    void init_data(q_info* details){
        ser_info.addr_images_in = details->addr_images_in;//organize data for use
        ser_info.rkey_images_in = details->rkey_images_in;//organize data for use
        ser_info.addr_images_out = details->addr_images_out;//organize data for use
        ser_info.rkey_images_out = details->rkey_images_out;//organize data for use
        blocks = details->blocks;
    }
    void init_mr(){
        mr_cpu_gpu_entry = ibv_reg_mr(pd, &cpu_gpu_entry,sizeof(cpu_to_gpu_entry),  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_cpu_gpu_entry) exit(1);//entries as memory region's
        mr_gpu_cpu_entry = ibv_reg_mr(pd, &gpu_cpu_entry,sizeof(gpu_to_cpu_entry),  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_gpu_cpu_entry)  exit(1);//entries as memory region's
    }

    inline void init_server_space(){
        for(int i = 0; i<OUTSTANDING_REQUESTS; i++){
            server_space[i] = ISFREE;
        }
    }

    void kill_q(){
        for(int i = 0; i < blocks; i++){
            cpu_to_gpu[i].~cli_q();
            gpu_to_cpu[i].~cli_q();
        }
    }

    void polling_wait(){
        struct ibv_wc completion; 
        int count = 0;
        while(count == 0) {
            count = ibv_poll_cq(cq, 1, &completion);
            if (count < 0) exit(1);
        }
        VERBS_WC_CHECK(completion);
    }

    inline void read_q_enq(cli_q<cpu_to_gpu_entry> *cur_q){
        post_rdma_read(&(cur_q->ci), sizeof(int), cur_q->mr_ci->lkey, (uint64_t)&(cur_q->addr)->ci, cur_q->rkey, wr_id++);
        polling_wait();
        post_rdma_read(&(cur_q->pi), sizeof(int), cur_q->mr_pi->lkey, (uint64_t)&(cur_q->addr)->pi, cur_q->rkey, wr_id++);
        polling_wait();
    }

    int find_empty_space(){
        int i=0;
        for(; i<OUTSTANDING_REQUESTS; i++){
            if(server_space[i] == ISFREE) {
                return i; 
            }
        }
        if(i == OUTSTANDING_REQUESTS){
            return false; 
        }
        return -1; //should not get here 
    }

    inline void read_q_deq(cli_q<gpu_to_cpu_entry> *cur_q){
        post_rdma_read(&(cur_q->pi), sizeof(int), (cur_q->mr_pi)->lkey, (uint64_t)&(cur_q->addr)->pi, cur_q->rkey, wr_id++);
        polling_wait();
        post_rdma_read(&(cur_q->ci), sizeof(int), (cur_q->mr_ci)->lkey, (uint64_t)&(cur_q->addr)->ci, cur_q->rkey, wr_id++);
        polling_wait();
    }

    int find_done_space(){
        int i=0;
        for( ; i<OUTSTANDING_REQUESTS; i++) {
            if(server_space[i] != gpu_cpu_entry.img_idx){
                continue;
            }else{
                server_space[i] = ISFREE;
                return i;
            }
        }
        if (i == OUTSTANDING_REQUESTS){
            return -1;
        }
        return 0;
    }

public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port) {
        q_info details;
        recv_over_socket(&details, sizeof(q_info));
        this->init_data(&details);
        cpu_to_gpu = (cli_q<cpu_to_gpu_entry>*)malloc(sizeof(cli_q<cpu_to_gpu_entry>)*blocks);
        gpu_to_cpu = (cli_q<gpu_to_cpu_entry>*)malloc(sizeof(cli_q<gpu_to_cpu_entry>)*blocks);
        for(int i = 0; i < blocks; i++){
            new (&cpu_to_gpu[i]) cli_q<cpu_to_gpu_entry>(&details.add_cpu_to_gpu[i], details.rkey_cpu_to_gpu, pd);
            new (&gpu_to_cpu[i]) cli_q<gpu_to_cpu_entry>(&details.add_gpu_to_cpu[i], details.rkey_gpu_to_cpu, pd);
        }
        this->init_mr();
        this->init_server_space();
        wr_id = 0;
    }

    ~client_queues_context() {
        bool kill = true;
        send_over_socket(&kill, sizeof(bool));
        ibv_dereg_mr(mr_cpu_gpu_entry); //deregister 
        ibv_dereg_mr(mr_gpu_cpu_entry); //deregister 
        ibv_dereg_mr(mr_images_in);    //deregister 
        ibv_dereg_mr(mr_images_out);   //deregister 
        this->kill_q();
        free(cpu_to_gpu);
        free(gpu_to_cpu);
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override {
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) exit(1);
        cli_info.lkey_images_in = mr_images_in->lkey;
        cli_info.addr_images_in = (uint64_t)mr_images_in->addr;
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override {
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) exit(1);
        cli_info.addr_images_out = (uint64_t)mr_images_out->addr;
        cli_info.lkey_images_out = mr_images_out->lkey;
    }


    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override {
        for(int i = 0; i<blocks; i++) { 
            cli_q<cpu_to_gpu_entry> *cur_q = &cpu_to_gpu[i];
            this->read_q_enq(&cpu_to_gpu[i]);
            int diff = cur_q->pi - cur_q->ci;
            if(diff == NSLOTS){
                return false;
            }else{
                int i= find_empty_space();
                server_space[i] = img_id;
                int offset = IMG_SZ*i;
                post_rdma_write(ser_info.addr_images_in + offset, IMG_SZ*sizeof(uchar), ser_info.rkey_images_in, img_in, cli_info.lkey_images_in, wr_id++, nullptr);
                polling_wait();
                cpu_gpu_entry.img_idx = img_id;
                cpu_gpu_entry.img_in = (uchar*)ser_info.addr_images_in + offset;
                cpu_gpu_entry.img_out = (uchar*)ser_info.addr_images_out + offset;
                post_rdma_write((uint64_t)&(cur_q->addr->data[cur_q->pi%NSLOTS]), sizeof(cpu_to_gpu_entry), cur_q->rkey, &cpu_gpu_entry, mr_cpu_gpu_entry->lkey, wr_id++, nullptr);
                polling_wait();
                cur_q->pi += 1;
                post_rdma_write((uint64_t)&(cur_q->addr->pi), sizeof(int), cur_q->rkey, &(cur_q->pi), (cur_q->mr_pi)->lkey, wr_id++, nullptr);
                polling_wait();
                return true;
            }
        }
        return false;
    }


    virtual bool dequeue(int *img_id) override {
        for(int i = 0; i<blocks; i++) { 
            cli_q<gpu_to_cpu_entry> *cur_q = &gpu_to_cpu[i];
            this->read_q_deq(&gpu_to_cpu[i]);
            if(cur_q->pi == cur_q->ci){
                return false;
            }
            else { 
                gpu_cpu_entry.img_idx = -1;
                post_rdma_read(&gpu_cpu_entry, sizeof(gpu_to_cpu_entry), mr_gpu_cpu_entry->lkey, (uint64_t)&cur_q->addr->data[cur_q->ci%NSLOTS], cur_q->rkey, wr_id++);
                polling_wait();
                int offset= IMG_SZ*this->find_done_space();
                post_rdma_read((uchar*)cli_info.addr_images_out + IMG_SZ*(gpu_cpu_entry.img_idx%N_IMAGES), IMG_SZ*sizeof(uchar), cli_info.lkey_images_out, ser_info.addr_images_out + offset, ser_info.rkey_images_out, wr_id++);
                polling_wait();
                cur_q->ci += 1;
                post_rdma_write((uint64_t)&(cur_q->addr)->ci, sizeof(int), cur_q->rkey, &cur_q->ci, cur_q->mr_ci->lkey, wr_id++, nullptr);
                polling_wait();
                *img_id = gpu_cpu_entry.img_idx;
                return true;
            }
        }
        return false;
    }

};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
        case MODE_RPC_SERVER:
            return std::make_unique<server_rpc_context>(tcp_port);
        case MODE_QUEUE:
            return std::make_unique<server_queues_context>(tcp_port);
        default:
            printf("Unknown mode.\n");
            exit(1);
    }
}

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
        case MODE_RPC_SERVER:
            return std::make_unique<client_rpc_context>(tcp_port);
        case MODE_QUEUE:
            return std::make_unique<client_queues_context>(tcp_port);
        default:
            printf("Unknown mode.\n");
            exit(1);
    }
}