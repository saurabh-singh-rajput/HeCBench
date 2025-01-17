#ifndef CUDA_PROFILER_H
#define CUDA_PROFILER_H

#include <nvml.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>    // Add this for sqrt, pow
#include <limits>   // Add this for numeric_limits
#include <ctime>    // Add this for time operations
#include <iomanip>

struct KernelConfig {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMemBytes;
    cudaStream_t stream;

    KernelConfig() : gridDim(1), blockDim(1), sharedMemBytes(0), stream(0) {}
    
    KernelConfig(dim3 grid, dim3 block, size_t shared = 0, cudaStream_t str = 0)
        : gridDim(grid), blockDim(block), sharedMemBytes(shared), stream(str) {}
};

class CUDAProfiler {
private:
    nvmlDevice_t device;
    bool initialized;
    unsigned int sampling_rate_ms;
    std::atomic<bool> is_sampling;
    std::thread sampling_thread;
    std::string csv_filename;
    bool output_to_csv;

    struct ProfilePoint {
        cudaEvent_t start;
        cudaEvent_t stop;
        std::vector<unsigned int> power_samples;
        std::vector<long long> timestamps;  // Timestamps for each sample
        float time_ms;
        std::string label;
        KernelConfig config;

        ProfilePoint() : start(nullptr), stop(nullptr), time_ms(0) {}
    };

    std::vector<ProfilePoint> measurements;

    unsigned int getCurrentPower() {
        unsigned int power;
        nvmlDeviceGetPowerUsage(device, &power);
        return power;
    }

    void powerSamplingThread(ProfilePoint& point) {
        // Increase sampling frequency and use microseconds for better precision
        const unsigned int sampling_rate_us = 100; // 100 microseconds
        auto start_time = std::chrono::steady_clock::now();
        
        while (is_sampling) {
            unsigned int power = getCurrentPower();
            if (power > 0) {  // Only record valid power readings
                point.power_samples.push_back(power);
                auto current_time = std::chrono::steady_clock::now();
                point.timestamps.push_back(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        current_time - start_time).count());
            }
            std::this_thread::sleep_for(std::chrono::microseconds(sampling_rate_us));
        }
    }

public:
    CUDAProfiler(unsigned int sampling_ms = 1, bool csv_output = false, 
                 const std::string& filename = "cuda_measurements.csv") 
        : initialized(false), sampling_rate_ms(sampling_ms), 
          is_sampling(false), output_to_csv(csv_output), 
          csv_filename(filename) {
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
            return;
        }
        
        result = nvmlDeviceGetHandleByIndex(0, &device);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
            nvmlShutdown();
            return;
        }
        
        initialized = true;
    }

    ~CUDAProfiler() {
        if (is_sampling && sampling_thread.joinable()) {
            is_sampling = false;
            sampling_thread.join();
        }
        if (initialized) {
            nvmlShutdown();
        }
    }

    
    void printProgress(int current, int total) {  
    if (current == 1 || current == total || current % 10 == 0) {  // Print only at start, every 10th config, and end
        std::cout << "Testing configuration " << current << "/" << total << "\n" << std::flush;
    }
     }
     
    void startMeasurement(const std::string& label, dim3 gridDim, dim3 blockDim, 
                         size_t sharedMem = 0, cudaStream_t stream = 0) {
        if (!initialized) return;

        measurements.emplace_back();
        ProfilePoint& point = measurements.back();
        point.label = label;
        point.config = KernelConfig(gridDim, blockDim, sharedMem, stream);
        
        cudaEventCreate(&point.start);
        cudaEventCreate(&point.stop);
        
        // Start power sampling thread
        is_sampling = true;
        sampling_thread = std::thread(&CUDAProfiler::powerSamplingThread, this, std::ref(point));
        
        cudaEventRecord(point.start);
    }

    void startMeasurement(const std::string& label, unsigned int gridX, unsigned int blockX) {
        startMeasurement(label, dim3(gridX), dim3(blockX));
    }

    void stopMeasurement() {
        if (!initialized || measurements.empty()) return;

        ProfilePoint& point = measurements.back();
        cudaEventRecord(point.stop);
        
        // Stop power sampling thread
        is_sampling = false;
        if (sampling_thread.joinable()) {
            sampling_thread.join();
        }
        
        cudaEventSynchronize(point.stop);
        cudaEventElapsedTime(&point.time_ms, point.start, point.stop);
        
        cudaEventDestroy(point.start);
        cudaEventDestroy(point.stop);
    }

    void printMeasurements() {
        if (!initialized) return;

        std::ofstream csv_file;
        if (output_to_csv) {
            bool write_header = !std::ifstream(csv_filename).good();
            csv_file.open(csv_filename, std::ios::app);
            
            if (write_header) {
                csv_file << "Run_ID,"
                        << "Kernel_Name,"
                        << "Thread_Config,"
                        << "Grid_X,"        
                        << "Grid_Y,"
                        << "Grid_Z,"
                        << "Block_X,"
                        << "Block_Y,"
                        << "Block_Z,"
                        << "Total_Blocks,"
                        << "Threads_per_Block,"
                        << "Total_Threads,"
                        << "Shared_Mem_Bytes,"
                        << "Execution_Time_ms,"
                        << "Avg_Power_mW,"
                        << "Energy_mJ,"
                        << "Min_Power_mW,"
                        << "Max_Power_mW,"
                        << "Stddev_Power_mW,"
                        << "Num_Power_Samples,"
                        << "Power_Sample_List\n";
            }
        }

        auto now = std::chrono::system_clock::now();
        auto run_id = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        for (const auto& point : measurements) {
            // Extract kernel name and thread configuration, removing brackets
            std::string kernel_name = point.label;
            std::string thread_config;
            
            // Find the last occurrence of '_(' and remove everything after it
            size_t config_start = kernel_name.find_last_of("_(");
            if (config_start != std::string::npos) {
                // Extract the thread configuration without brackets
                thread_config = point.label.substr(config_start + 2);
                // Remove the closing bracket if it exists
                if (!thread_config.empty() && thread_config.back() == ')') {
                    thread_config.pop_back();
                }
                // Get the clean kernel name
                kernel_name = point.label.substr(0, config_start);
            }
            else {
                kernel_name = point.label;
                thread_config = "unknown";
            }

            // Calculate metrics
            unsigned long total_blocks = point.config.gridDim.x * 
                                    point.config.gridDim.y * 
                                    point.config.gridDim.z;
            
            unsigned long threads_per_block = point.config.blockDim.x * 
                                            point.config.blockDim.y * 
                                            point.config.blockDim.z;
            
            unsigned long total_threads = total_blocks * threads_per_block;

            // Calculate power statistics
            double avg_power = 0.0;
            double min_power = 0.0;
            double max_power = 0.0;
            double stddev_power = 0.0;
            std::string power_samples_str = "";

            if (!point.power_samples.empty()) {
                unsigned long long total_power = 0;
                min_power = point.power_samples[0];
                max_power = point.power_samples[0];
                
                for (unsigned int power : point.power_samples) {
                    total_power += power;
                    min_power = std::min(min_power, (double)power);
                    max_power = std::max(max_power, (double)power);
                    power_samples_str += std::to_string(power) + "|";
                }
                
                avg_power = total_power / (double)point.power_samples.size();

                // Calculate standard deviation
                double variance = 0.0;
                for (unsigned int power : point.power_samples) {
                    variance += pow(power - avg_power, 2);
                }
                if (point.power_samples.size() > 1) {
                    variance /= (point.power_samples.size() - 1);
                    stddev_power = sqrt(variance);
                }
            }

            double energy_mj = (point.time_ms/1000.0) * avg_power;

            if (output_to_csv) {
                csv_file << run_id << ","
                        << kernel_name << ","
                        << thread_config << ","
                        << point.config.gridDim.x << ","
                        << point.config.gridDim.y << ","
                        << point.config.gridDim.z << ","
                        << point.config.blockDim.x << ","
                        << point.config.blockDim.y << ","
                        << point.config.blockDim.z << ","
                        << total_blocks << ","
                        << threads_per_block << ","
                        << total_threads << ","
                        << point.config.sharedMemBytes << ","
                        << std::fixed << std::setprecision(6) << point.time_ms << ","
                        << std::fixed << std::setprecision(2) << avg_power << ","
                        << std::fixed << std::setprecision(6) << energy_mj << ","
                        << std::fixed << std::setprecision(2) << min_power << ","
                        << std::fixed << std::setprecision(2) << max_power << ","
                        << std::fixed << std::setprecision(2) << stddev_power << ","
                        << point.power_samples.size() << ","
                        << power_samples_str << "\n";
            }

            std::cout << "\nKernel: " << point.label << std::endl;
        }

        if (output_to_csv) {
            csv_file.close();
            std::cout << "Measurements saved to: " << csv_filename << std::endl;
        }
}

    void reset() {
        if (is_sampling && sampling_thread.joinable()) {
            is_sampling = false;
            sampling_thread.join();
        }
        measurements.clear();
    }

    // Get raw measurements for external analysis
    std::vector<std::pair<std::vector<long long>, std::vector<unsigned int>>> 
    getRawPowerSamples() {
        std::vector<std::pair<std::vector<long long>, std::vector<unsigned int>>> results;
        for (const auto& point : measurements) {
            results.emplace_back(point.timestamps, point.power_samples);
        }
        return results;
    }
};

#endif // CUDA_PROFILER_H