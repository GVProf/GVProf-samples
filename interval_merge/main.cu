#include <cub/cub.cuh>
#include <chrono>
#include <set>

#include <iostream>
#include <algorithm>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>

#define THREADS 256
#define ITEMS 16
#define N (ITEMS * THREADS)
#define KEY uint64_t
#define ITER 1000
#define WARP_SIZE 32

__device__ __forceinline__ uint32_t bfe(uint32_t source, uint32_t bit_index) {
	uint32_t bit;
	asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(bit) : "r"((uint32_t) source), "r"(bit_index), "r"(1));
	return bit;
}

__device__ __forceinline__ uint32_t bfind(uint32_t source) {
	uint32_t bit_index;
	asm volatile("bfind.u32 %0, %1;" : "=r"(bit_index) : "r"((uint32_t) source));
	return bit_index;
}

__device__ __forceinline__ uint32_t fns(uint32_t source, uint32_t base_index) {
	uint32_t bit_index;
	asm volatile("fns.b32 %0, %1, %2, %3;" : "=r"(bit_index) : "r"(source), "r"(base_index), "r"(1));
	return bit_index;
}

__device__ __forceinline__ uint64_t comparator(const uint64_t x, uint32_t lane_mask, bool dir, uint32_t mask) {
	auto y = __shfl_xor_sync(mask, x, lane_mask);
	return x < y == dir ? y : x;
}

__device__ __forceinline__ uint32_t get_lane_id() {
	uint32_t lane_id;
	asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
	return lane_id;
}

__launch_bounds__(WARP_SIZE, 1)
__global__
void warp_merge_interval
(
 KEY *d_in,
 KEY *d_out
)
{
  uint64_t x = 0xFFFFFFFFFFFFFFFF;

  if (threadIdx.x < 8) {
    x = d_in[threadIdx.x];

    auto mask = __activemask();
    auto lane_id = get_lane_id();
    uint32_t first_laneid = __ffs(mask) - 1;

    printf("%d\n", x);

    x = comparator(x, 1, bfe(lane_id, 1) ^ bfe(lane_id, 0), mask); // A, sorted sequences of length 2
    x = comparator(x, 2, bfe(lane_id, 2) ^ bfe(lane_id, 1), mask); // B
    x = comparator(x, 1, bfe(lane_id, 2) ^ bfe(lane_id, 0), mask); // C, sorted sequences of length 4
    x = comparator(x, 4, bfe(lane_id, 3) ^ bfe(lane_id, 2), mask); // D
    x = comparator(x, 2, bfe(lane_id, 3) ^ bfe(lane_id, 1), mask); // E
    x = comparator(x, 1, bfe(lane_id, 3) ^ bfe(lane_id, 0), mask); // F, sorted sequences of length 8
    x = comparator(x, 8, bfe(lane_id, 4) ^ bfe(lane_id, 3), mask); // G
    x = comparator(x, 4, bfe(lane_id, 4) ^ bfe(lane_id, 2), mask); // H
    x = comparator(x, 2, bfe(lane_id, 4) ^ bfe(lane_id, 1), mask); // I
    x = comparator(x, 1, bfe(lane_id, 4) ^ bfe(lane_id, 0), mask); // J, sorted sequences of length 16
    x = comparator(x, 16, bfe(lane_id, 4), mask); // K
    x = comparator(x, 8, bfe(lane_id, 3), mask); // L
    x = comparator(x, 4, bfe(lane_id, 2), mask); // M
    x = comparator(x, 2, bfe(lane_id, 1), mask); // N
    x = comparator(x, 1, bfe(lane_id, 0), mask); // O, sorted sequences of length 32

    printf("%d\n", x);

#if 0
    uint32_t y = __shfl_up_sync(mask, x, 1);
    if (y + 1 != x || lane_id == first_laneid) {
      y = 1;
    } else {
      y = 0;
    }
    uint32_t b = __ballot_sync(mask, y);

    // find the end position
    uint32_t p = fns(b, lane_id + 1);
    if (p == 0xFFFFFFFF) {
      p = 31;
    }
    auto z = __shfl_sync(mask, x, p);

    if (y == 1) {
      printf("%d\n", lane_id);
      // Push an interval
      d_out[threadIdx.x] = z;
    }
#endif
    d_out[threadIdx.x] = x;
  }
}


__launch_bounds__(THREADS, 1)
__global__
void gpu_merge_interval
(
 KEY *d_in,
 KEY *d_out
)
{
	enum { TILE_SIZE = THREADS * ITEMS };
	// Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
	typedef cub::BlockLoad<KEY, THREADS, ITEMS, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
  // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef cub::BlockStore<KEY, THREADS, ITEMS, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
	// Specialize BlockRadixSort type for our thread block
	typedef cub::BlockRadixSort<KEY, THREADS, ITEMS, int> BlockRadixSortT;
  // Specialize BlockScan type for our thread block
  typedef cub::BlockScan<int, THREADS> BlockScanT;
	// Shared memory
	__shared__ union TempStorage
	{
		typename BlockLoadT::TempStorage        load;
    typename BlockStoreT::TempStorage       store;
		typename BlockRadixSortT::TempStorage   sort;
    typename BlockScanT::TempStorage        scan;
	} temp_storage;

	// Per-thread tile items
	KEY items[ITEMS];
  int interval_start[ITEMS];
  int interval_end[ITEMS];
  int interval_start_index[ITEMS];
  int interval_end_index[ITEMS];

	// Load items into a blocked arrangement
	BlockLoadT(temp_storage.load).Load(d_in, items);

  for (size_t i = 0; i < ITEMS / 2; ++i) {
    items[i * 2 + 1] += 1;
  }

	// Barrier for smem reuse
	__syncthreads();

  for (size_t i = 0; i < ITEMS / 2; ++i) {
    interval_start[i * 2] = 1;
    interval_start[i * 2 + 1] = -1;
    interval_end[i * 2] = 0;
    interval_end[i * 2 + 1] = 0;
    interval_start_index[i * 2] = 0;
    interval_start_index[i * 2 + 1] = 0;
    interval_end_index[i * 2] = 0;
    interval_end_index[i * 2 + 1] = 0;
  }

	// Sort keys
	BlockRadixSortT(temp_storage.sort).Sort(items, interval_start);
  __syncthreads();

  // Get start/end marks
  BlockScanT(temp_storage.scan).InclusiveSum(interval_start, interval_start);
  __syncthreads();

  for (size_t i = 0; i < ITEMS; ++i) {
    if (interval_start[i] == 1) {
      // do nothing
    } else if (interval_start[i] == 0) {
      interval_end[i] = 1;
    } else {
      interval_start[i] = 0;
    }
  }
  // Get interval start index
  BlockScanT(temp_storage.scan).InclusiveSum(interval_start, interval_start_index);
  __syncthreads();

  // Get interval end index
  BlockScanT(temp_storage.scan).InclusiveSum(interval_end, interval_end_index);
  __syncthreads();

  // Put indices in the corresponding slots
  for (size_t i = 0; i < ITEMS; ++i) {
    if (interval_start[i] == 1) {
      d_out[(interval_start_index[i] - 1) * 2] = items[i];
    }
    if (interval_end[i] == 1) {
      d_out[(interval_end_index[i] - 1) * 2 + 1] = items[i] - 1;
    }
  }
}

void init(KEY *data) {
	for (size_t i = 0; i < N/2; ++i) {
		data[i * 2] = i;
		data[i * 2 + 1] = i + 1;
	} 
}

struct MemoryRange {
  uint64_t start;
  uint64_t end;

  MemoryRange() = default;

  MemoryRange(uint64_t start, uint64_t end) : start(start), end(end) {}

  bool operator<(const MemoryRange &other) const { return start < other.start; }
};

void interval_merge(std::set<MemoryRange> &memory, MemoryRange &memory_range) {
  bool delete_left = false;
  bool delete_right = false;

  auto liter = memory.upper_bound(memory_range);
  if (liter != memory.begin()) {
		--liter;
    if (liter->end >= memory_range.start) {
      if (liter->end < memory_range.end) {
        // overlap and not covered
        delete_left = true;
      } else {
        return;
      }
    }
  }

  auto riter = memory.lower_bound(memory_range);
  if (riter != memory.end()) {
    if (riter->start <= memory_range.end) {
      if (riter->start > memory_range.start) {
        // overlap and not covered
        delete_right = true;
      } else {
        return;
      }
    }
  }

  auto start = memory_range.start;
  auto end = memory_range.end;
  if (delete_left) {
    start = liter->start;
    memory.erase(liter);
  }
  if (delete_right) {
    end = riter->end;
    memory.erase(riter);
  }
  memory.insert(MemoryRange(start, end));
}

void cpu_interval_merge(KEY *data) {
	std::set<MemoryRange> memory;
	for (size_t i = 0; i < N/2; ++i) {
    auto memory_range = MemoryRange(data[i*2], data[i*2 + 1]);
		interval_merge(memory, memory_range);
	}

	size_t i = 0;
	for (auto memory_range : memory) {
		data[i++] = memory_range.start;
		data[i++] = memory_range.end;
	}
}

void check(KEY *data) {
  std::cout << "Interval: [" << data[0] << ", " << data[1] << "]" << std::endl;
}

int main() {
	KEY *input = new KEY[N];
	KEY *output = new KEY[N];

	init(input);

	// CPU
	auto start = std::chrono::steady_clock::now();
	for (size_t i = 0; i < ITER; ++i) {
		std::copy(input, input + N, output);
    cpu_interval_merge(output);
	}
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cout << "CPU elapsed time: " << elapsed_seconds.count() << "s\n";

  check(output);

	KEY *d_input = NULL;
	KEY *d_output = NULL;
	cudaMalloc(&d_input, N * sizeof(KEY));
	cudaMalloc(&d_output, N * sizeof(KEY));
	cudaMemcpy(d_input, input, N * sizeof(KEY), cudaMemcpyHostToDevice);

	// GPU
	start = std::chrono::steady_clock::now();
	for (size_t i = 0; i < ITER; ++i) {
		cudaMemcpy(d_output, d_input, N * sizeof(KEY), cudaMemcpyDeviceToDevice);
		gpu_merge_interval<<<1, THREADS>>>(d_output, d_output);
	}
	end = std::chrono::steady_clock::now();

	cudaMemcpy(output, d_output, N * sizeof(KEY), cudaMemcpyDeviceToHost);

	elapsed_seconds = end-start;

	std::cout << "GPU elapsed time: " << elapsed_seconds.count() << "s\n";

  check(output);

  // GPU warp 
  for (size_t i = 0; i < 32; ++i) {
    input[i] = 32 - i - 1;
  }
  input[30] = 45;
  input[31] = 46;

	cudaMemcpy(d_output, input, 32 * sizeof(KEY), cudaMemcpyHostToDevice);
  warp_merge_interval<<<1, 32>>>(d_output, d_output);
	cudaMemcpy(output, d_output, 32 * sizeof(KEY), cudaMemcpyDeviceToHost);

  std::cout << "Sort: ";
  for (size_t i = 0; i < 32; ++i) {
    std::cout << output[i] << ", ";
  }
  std::cout << std::endl;
	
	return 0;
}

