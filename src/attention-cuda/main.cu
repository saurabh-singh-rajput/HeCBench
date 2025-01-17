#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include "kernels.h"
#include "reference.h"
#include <nvml.h>
#include "cuda_profiler.h"
#include "thread_config.h"

float* attention_device(const float* key, const float* value, const float* query,
                        const int n, const int d, const int impl_num, const int repeat)
{
  // input
  float *d_key;
  cudaMalloc((void**)&d_key, n * d * sizeof(float));
  cudaMemcpy(d_key, key, n * d * sizeof(float), cudaMemcpyHostToDevice);

  float *d_value;
  cudaMalloc((void**)&d_value, n * d * sizeof(float));
  cudaMemcpy(d_value, value, n * d * sizeof(float), cudaMemcpyHostToDevice);

  float *d_query;
  cudaMalloc((void**)&d_query, d * sizeof(float));
  cudaMemcpy(d_query, query, d * sizeof(float), cudaMemcpyHostToDevice);

  // intermediate
  float *d_dot_product;
  cudaMalloc((void**)&d_dot_product, n * sizeof(float));

  float *d_exp_sum;
  cudaMalloc((void**)&d_exp_sum, sizeof(float));

  // result
  float *output = (float*) malloc (d * sizeof(float));
  float *d_output;
  cudaMalloc((void**)&d_output, d * sizeof(float));

  cudaDeviceSynchronize();

  CUDAProfiler profiler(1, true, "measurements.csv");;

  if (impl_num == 2) {
    // Initialize thread configuration generator
    auto config_gen = ThreadConfigs::getStandardConfigs();
    int total_configs = config_gen.getTotalConfigCount() * repeat;
    printf("Total configurations: %d\n", total_configs); 
    int current_config = 0;
    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
            // For each thread count configuration set
            for (const auto& config_set : config_gen.getAllConfigs()) {
                current_config++;
                profiler.printProgress(current_config, total_configs);
                for (const auto& config : config_set) {
                    cudaMemset(d_exp_sum, 0, 4);
                    
                    // Configure dimensions for kernel1_warpReduce
                    dim3 block1 = config.getBlockDim();
                    int sm_count;
                    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
                    int warps_per_sm = 32; // This varies by GPU architecture
                    int target_blocks = sm_count * warps_per_sm;
                    dim3 grid1((n + block1.x - 1) / block1.x);
                    // Adjust grid size to be multiple of SM count
                    grid1.x = ((grid1.x + target_blocks - 1) / target_blocks) * target_blocks;
                    // dim3 grid1((n + block1.x - 1) / block1.x, 1);
                    
                    std::string label1 = ThreadConfigGenerator::getKernelLabel(
                        "kernel1_warpReduce", config);
                    
                    profiler.startMeasurement(label1, grid1, block1);
                    kernel1_warpReduce<<<grid1, block1>>>(
                        d_key, d_query, d_dot_product, d_exp_sum, n, d);
                    profiler.stopMeasurement();

                    // Configure dimensions for kernel2_blockReduce
                    dim3 block2 = config.getBlockDim();
                    dim3 grid2((d + block2.x - 1) / block2.x, 1);
                    
                    std::string label2 = ThreadConfigGenerator::getKernelLabel(
                        "kernel2_blockReduce", config);
                    
                    profiler.startMeasurement(label2, grid2, block2);
                    kernel2_blockReduce<<<grid2, block2>>>(
                        d_exp_sum, d_dot_product, d_value, d_output, n, d);
                    profiler.stopMeasurement();
                }
            }
        }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else if (impl_num == 2) {

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      cudaMemset(d_exp_sum, 0, 4);
      kernel1_warpReduce<<<(n+7)/8, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      kernel2_warpReduce<<<(d+7)/8, 256>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else if (impl_num == 1) {

    auto config_gen = ThreadConfigs::getStandardConfigs();
    int total_configs = config_gen.getTotalConfigCount() * repeat;
    printf("Total configurations: %d\n", total_configs); 
    int current_config = 0;
    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
            for (const auto& config_set : config_gen.getAllConfigs()) {
                current_config++;
                profiler.printProgress(current_config, total_configs);
                for (const auto& config : config_set) {
                    cudaMemset(d_exp_sum, 0, 4);
                    
                    dim3 block = config.getBlockDim();
                    // dim3 grid1(n, 1);
                    // dim3 grid2(d, 1);
                    
                    dim3 grid1((n + block.x - 1) / block.x, 1);
                    dim3 grid2((d + block.x - 1) / block.x, 1);

                    std::string label3 = ThreadConfigGenerator::getKernelLabel(
                        "kernel1_blockReduce", config);
                    profiler.startMeasurement(label3, grid1, block);
                    kernel1_blockReduce<<<grid1, block>>>(
                        d_key, d_query, d_dot_product, d_exp_sum, n, d);
                    profiler.stopMeasurement();

                    std::string label4 = ThreadConfigGenerator::getKernelLabel(
                        "kernel2_blockReduce", config);
                    profiler.startMeasurement(label4, grid2, block);
                    kernel2_blockReduce<<<grid2, block>>>(
                        d_exp_sum, d_dot_product, d_value, d_output, n, d);
                    profiler.stopMeasurement();
                }
            }
        }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else {
    auto config_gen = ThreadConfigs::getStandardConfigs();
    int total_configs = config_gen.getTotalConfigCount() * repeat;
    printf("Total configurations: %d\n", total_configs); 
    int current_config = 0;
    float *d_score;
    cudaMalloc((void**)&d_score, n * sizeof(float));

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
            for (const auto& config_set : config_gen.getAllConfigs()) {
                current_config++;
                profiler.printProgress(current_config, total_configs);
                for (const auto& config : config_set) {
                    cudaMemset(d_exp_sum, 0, 4);
                    
                    dim3 block = config.getBlockDim();
                    dim3 grid1((n + block.x - 1) / block.x, 1);
                    dim3 grid2((d + block.x - 1) / block.x, 1);
                    
                    std::string label5 = ThreadConfigGenerator::getKernelLabel("kernel1", config);
                    profiler.startMeasurement(label5, grid1, block);
                    kernel1<<<grid1, block>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
                    profiler.stopMeasurement();

                    std::string label6 = ThreadConfigGenerator::getKernelLabel("kernel2", config);
                    profiler.startMeasurement(label6, grid1, block);
                    kernel2<<<grid1, block>>>(d_exp_sum, d_dot_product, d_score, n);
                    profiler.stopMeasurement();

                    std::string label7 = ThreadConfigGenerator::getKernelLabel("kernel3", config);
                    profiler.startMeasurement(label7, grid2, block);
                    kernel3<<<grid2, block>>>(d_score, d_value, d_output, n, d);
                    profiler.stopMeasurement();
                }
            }
        }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
    cudaFree(d_score);
  }

  profiler.printMeasurements();

  cudaMemcpy(output, d_output, d * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_value);
  cudaFree(d_output);
  cudaFree(d_key);
  cudaFree(d_dot_product);
  cudaFree(d_exp_sum);
  return output;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <rows> <columns> <implementation> <repeat>\n", argv[0]);
    printf("implementation 0: naive\n");
    printf("implementation 1: fused kernels with block reduce\n");
    printf("implementation 2: fused kernels with warp reduce\n");
    printf("implementation 3: fused kernels with mixed reduce\n");
    return 1;
  }
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int r = atoi(argv[4]);

  // input
  float* key = (float*) malloc (n * d * sizeof(float));
  float* value = (float*) malloc (n * d * sizeof(float));
  float* query = (float*) malloc (d * sizeof(float));

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  for (int i = 0; i < n * d; i++) {
    key[i] = dist(gen);
    value[i] = dist(gen);
    query[i % d] = dist(gen);
  }

  float* hout = attention_host(key, value, query, n, d);

  float* dout = attention_device(key, value, query, n, d, k, r);

  float rmse = 0;
  for (int i = 0; i < d; i++) {
    rmse += (hout[i] - dout[i]) * (hout[i] - dout[i]);
  }
  printf("RMSE = %f\n", sqrtf(rmse / d));

  free(key);
  free(value);
  free(query);
  free(dout);
  free(hout);
  return 0;
}
