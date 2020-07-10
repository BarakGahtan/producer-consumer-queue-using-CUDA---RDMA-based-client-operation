/* A solution for the queues part of hw2. Feel free to replace with your own or
 * to change it to your needs. */

#include "ex3.h"
#include "ex2.h"

#include <cassert>

#include <cuda/atomic>

using cuda::memory_order_relaxed;
using cuda::memory_order_acquire;
using cuda::memory_order_release;

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;

    for (int stride = 1; stride < min(blockDim.x, arr_size); stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__device__ void gpu_process_image_helper(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ uchar map[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < IMG_SZ; i += blockDim.x) {
        uchar c = in[i];
        atomicAdd(&histogram[c], 1);
    }

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        float map_value = float(histogram[tid]) / IMG_SZ;
        map[tid] = ((uchar)(N_COLORS * map_value)) * (256 / N_COLORS);
    }

    __syncthreads();

    for (int i = tid; i < IMG_SZ; i += blockDim.x) {
        out[i] = map[in[i]];
    }
}

__global__ void process_image_kernel(uchar *all_in, uchar *all_out)
{
    uchar *in = &all_in[blockIdx.x * IMG_SZ];
    uchar *out = &all_out[blockIdx.x * IMG_SZ];

    gpu_process_image_helper(in, out);
}


// Code assumes it is a power of two
#define NSLOTS 16

struct cpu_to_gpu_entry {
    int img_idx;
    uchar *img_in, *img_out;
};

struct gpu_to_cpu_entry {
    int img_idx;
};

template <typename entry_type>
struct queue
{
    using entry = entry_type;

    entry data[NSLOTS];
    cuda::atomic<int> pi;
    cuda::atomic<int> ci;
    cuda::atomic<bool> kill;

    queue() : pi(0), ci(0), kill(false) {}

    __host__ __device__ bool empty()
    {
        return pi.load(memory_order_acquire) == ci.load(memory_order_relaxed);
    }

    __host__ __device__ entry* peek()
    {
        return &data[ci & (NSLOTS - 1)];
    }

    __host__ __device__ int wraparound_slots_inc(int last)
    {
	return (last + 1);
    }

    __host__ __device__ void pop()
    {
        auto cur_ci = ci.load(memory_order_relaxed);
        ci.store(wraparound_slots_inc(cur_ci), memory_order_release);
    }

    __host__ __device__ bool full()
    {
        auto cur_pi = pi.load(memory_order_relaxed);
        auto cur_ci = ci.load(memory_order_acquire);
        return (cur_pi - cur_ci) == NSLOTS;
    }

    __host__ __device__ entry* next()
    {
        return &data[pi & (NSLOTS - 1)];
    }

    __host__ __device__ void push()
    {
        auto cur_pi = pi.load(memory_order_relaxed);
        pi.store(wraparound_slots_inc(cur_pi), memory_order_release);
    }
};

__global__ void gpu_process_image_consumer(queue<cpu_to_gpu_entry> *cpu_to_gpu, queue<gpu_to_cpu_entry> *gpu_to_cpu) {
    auto & h2g = cpu_to_gpu[blockIdx.x];
    auto & g2h = gpu_to_cpu[blockIdx.x];

    int tid = threadIdx.x;
    __shared__ cpu_to_gpu_entry entry;
    __shared__ bool kill;

    if (tid == 0)
        kill = false;

    while (true) {
        if (tid == 0) {
            while (h2g.empty()) {
		if (h2g.kill.load(memory_order_relaxed)) {
                    kill = true;
		    break;
		}
            }
            entry = *h2g.peek();
            dbg_printf("[%d] got image (%d). pi = %d, ci = %d\n", blockIdx.x, entry.img_idx, h2g.pi.load(), h2g.ci.load());
        }
        __syncthreads();
	if (kill) {
	    if (tid == 0)
                dbg_printf("[%d:%d] got kill\n", blockIdx.x, threadIdx.x);
	    return;
	}

        if (tid == 0) {
            h2g.pop();
            dbg_printf("[%d] popped image. pi = %d, ci = %d\n", blockIdx.x, h2g.pi.load(), h2g.ci.load());
        }

        gpu_process_image_helper(entry.img_in, entry.img_out);

        __syncthreads();

        if (tid == 0) {
            while (g2h.full()) ;
            auto out_entry = g2h.next(); // must have one?
            out_entry->img_idx = entry.img_idx;
            dbg_printf("[%d] pushing image (%d). pi = %d, ci = %d\n", blockIdx.x, entry.img_idx, g2h.pi.load(), g2h.ci.load());
        }

        __syncthreads();

        if (tid == 0) {
            g2h.push();
            dbg_printf("[%d] pushed image. pi = %d, ci = %d\n", blockIdx.x, g2h.pi.load(), g2h.ci.load());
        }
    }
}

class queues_gpu_context : public gpu_image_processing_context
{
private:
    // TODO define queue server context (memory buffers, etc...)
    int blocks;
    char *queue_buffer;
    queue<cpu_to_gpu_entry> *cpu_to_gpu;
    queue<gpu_to_cpu_entry> *gpu_to_cpu;
    int next_block = 0;

public:
    explicit queues_gpu_context(int threads) :
        blocks(calc_blocks(threads))
    {
        // TODO initialize host state
        CUDA_CHECK(cudaMallocHost(&queue_buffer, sizeof(*cpu_to_gpu) * blocks + sizeof(*gpu_to_cpu) * blocks));
        
        cpu_to_gpu = new (queue_buffer) queue<cpu_to_gpu_entry>[blocks];
        gpu_to_cpu = new (queue_buffer + sizeof(queue<cpu_to_gpu_entry>[blocks])) queue<gpu_to_cpu_entry>[blocks];

        // TODO launch GPU producer-consumer kernel with given number of threads
        gpu_process_image_consumer<<<blocks, threads>>>(cpu_to_gpu, gpu_to_cpu);
    }

    ~queues_gpu_context() override
    {
        // TODO free resources allocated in constructor
	for (int b = 0; b < blocks; ++b)
	    cpu_to_gpu[b].kill.store(true, memory_order_relaxed);
        CUDA_CHECK(cudaDeviceSynchronize());
	cpu_to_gpu->~queue<cpu_to_gpu_entry>();
	gpu_to_cpu->~queue<gpu_to_cpu_entry>();
	CUDA_CHECK(cudaFreeHost(queue_buffer));
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        auto &next = cpu_to_gpu[img_id % blocks]; // find_next_queue(cpu_to_gpu, blocks);
        if (next.full())
            return false;

        dbg_printf("enqueued img id: %d\n", img_id);
        auto *entry = next.next();
        entry->img_idx = img_id;
        entry->img_in = img_in;
        entry->img_out = img_out;
        next.push();
        return true;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        int block = next_block;
        for (int i = 0; i < blocks; ++i, ++block)
        {
            if (block >= blocks)
                block -= blocks;
            if (!gpu_to_cpu[block].empty()) {
                auto *entry = gpu_to_cpu[block].peek();
                *img_id = entry->img_idx;
                dbg_printf("[CPU] got image %d\n", *img_id);
                gpu_to_cpu[block].pop();
                dbg_printf("block %d i %d next_block %d\n", block, i, next_block);
                next_block = block - 1;
                if (next_block < 0)
                    next_block += blocks;
                return true;
            }
        }

        return false;
    }

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
};

std::unique_ptr<gpu_image_processing_context> create_queues_server(int threads)
{
    return std::make_unique<queues_gpu_context>(threads);
}
