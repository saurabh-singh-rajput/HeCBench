#ifndef THREAD_CONFIG_H
#define THREAD_CONFIG_H

#include <string>
#include <vector>
#include <cuda_runtime.h>

struct ThreadConfig {
    int x;
    int y;
    int total() const { return x * y; }
    
    // Constructor for easy initialization
    ThreadConfig(int x_, int y_) : x(x_), y(y_) {}
    
    // Get block and grid dimensions
    dim3 getBlockDim() const { return dim3(x, y, 1); }
    
    dim3 getGridDim(int n, int d) const {
        return dim3((n + x - 1) / x, (d + y - 1) / y, 1);
    }
    
    // Get configuration as string (for labeling)
    std::string toString() const {
        return "(" + std::to_string(x) + "," + std::to_string(y) + ")";
    }
};

class ThreadConfigGenerator {
private:
    std::vector<std::vector<ThreadConfig>> all_configs;
    
    void generateConfigsForSize(int total_threads) {
        std::vector<ThreadConfig> configs;
        for (int x = total_threads; x >= 1; x--) {
            if (total_threads % x == 0) {
                int y = total_threads / x;
                configs.emplace_back(x, y);
            }
        }
        all_configs.push_back(configs);
    }

public:
    // Initialize with different thread counts to test
    ThreadConfigGenerator(const std::vector<int>& thread_counts) {
        for (int count : thread_counts) {
            generateConfigsForSize(count);
        }
    }
    
    // Get configurations for a specific thread count
    const std::vector<ThreadConfig>& getConfigs(int thread_count) const {
        for (size_t i = 0; i < all_configs.size(); i++) {
            if (!all_configs[i].empty() && 
                all_configs[i][0].total() == thread_count) {
                return all_configs[i];
            }
        }
        // Return first config set if no match (you might want to handle this differently)
        return all_configs[0];
    }
    
    // Get all configurations
    const std::vector<std::vector<ThreadConfig>>& getAllConfigs() const {
        return all_configs;
    }
    
    // Generate kernel label
    static std::string getKernelLabel(const std::string& kernel_name, 
                                    const ThreadConfig& config) {
        return kernel_name + "_" + config.toString();
    }
};

// Predefined common configurations
namespace ThreadConfigs {
    inline ThreadConfigGenerator getStandardConfigs() {
        return ThreadConfigGenerator({256, 512, 1024});
    }
    
    inline ThreadConfigGenerator getExtendedConfigs() {
        return ThreadConfigGenerator({256, 512, 1024, 2048});
    }
    
    // Get a specific set of configurations
    inline ThreadConfigGenerator getCustomConfigs(const std::vector<int>& thread_counts) {
        return ThreadConfigGenerator(thread_counts);
    }
}

#endif // THREAD_CONFIG_H