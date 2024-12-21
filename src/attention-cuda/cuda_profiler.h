#ifndef CUDA_PROFILER_H
#define CUDA_PROFILER_H

#include <nvml.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

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
        while (is_sampling) {
            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count();
            
            unsigned int power = getCurrentPower();
            point.power_samples.push_back(power);
            point.timestamps.push_back(timestamp);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate_ms));
        }
    }

public:
    CUDAProfiler(unsigned int sampling_ms = 1) 
        : initialized(false), sampling_rate_ms(sampling_ms), is_sampling(false) {
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

        std::cout << "\n=== CUDA Kernel Measurements ===\n";
        for (const auto& point : measurements) {
            std::cout << "\nKernel: " << point.label << std::endl;
            
            // Configuration output
            std::cout << "Grid Dim: (" 
                      << point.config.gridDim.x << ", "
                      << point.config.gridDim.y << ", "
                      << point.config.gridDim.z << ")" << std::endl;
            
            std::cout << "Block Dim: (" 
                      << point.config.blockDim.x << ", "
                      << point.config.blockDim.y << ", "
                      << point.config.blockDim.z << ")" << std::endl;
            
            unsigned long total_threads = 
                (unsigned long)point.config.gridDim.x * 
                point.config.gridDim.y * 
                point.config.gridDim.z * 
                point.config.blockDim.x * 
                point.config.blockDim.y * 
                point.config.blockDim.z;
            
            std::cout << "Total Blocks: " 
                      << (point.config.gridDim.x * 
                          point.config.gridDim.y * 
                          point.config.gridDim.z) << std::endl;
            
            std::cout << "Threads per Block: " 
                      << (point.config.blockDim.x * 
                          point.config.blockDim.y * 
                          point.config.blockDim.z) << std::endl;
            
            std::cout << "Total Threads: " << total_threads << std::endl;
            
            if (point.config.sharedMemBytes > 0) {
                std::cout << "Shared Memory: " << point.config.sharedMemBytes << " bytes" << std::endl;
            }
            
            std::cout << "Stream: " << point.config.stream << std::endl;
            
            // Performance metrics
            std::cout << "Execution time: " << point.time_ms << " ms" << std::endl;
            std::cout << "Number of power samples: " << point.power_samples.size() << std::endl;
            
            // Calculate average power
            double avg_power = 0;
            if (!point.power_samples.empty()) {
                unsigned long long total_power = 0;
                for (unsigned int power : point.power_samples) {
                    total_power += power;
                }
                avg_power = total_power / (double)point.power_samples.size();
            }
            
            std::cout << "Average power: " << avg_power << " mW" << std::endl;
            std::cout << "Energy consumed: " << (point.time_ms/1000.0) * avg_power << " mJ" << std::endl;
            
            // Optional: Print detailed power samples
            std::cout << "\nPower samples (timestamp, power):" << std::endl;
            for (size_t i = 0; i < point.power_samples.size(); ++i) {
                std::cout << point.timestamps[i] << "ms: " 
                         << point.power_samples[i] << "mW" << std::endl;
            }
        }
        std::cout << "==============================\n" << std::endl;
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