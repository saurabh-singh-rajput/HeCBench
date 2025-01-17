import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

class BenchmarkAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
        # First convert Thread_Config to string type
        self.df['Thread_Config'] = self.df['Thread_Config'].astype(str)
        
        try:
            # Extract thread dimensions from Thread_Config
            self.df[['Thread_X', 'Thread_Y']] = self.df['Thread_Config'].str.split(',', expand=True).astype(int)
        except:
            print("Warning: Could not split Thread_Config into X,Y components")
            print("Thread_Config values:", self.df['Thread_Config'].unique())
            # Create dummy values if splitting fails
            self.df['Thread_X'] = self.df['Total_Threads'] 
            self.df['Thread_Y'] = 1
        
        # Setup style
        plt.style.use('default')
        self.colors = sns.color_palette("husl", 8)

    def save_plot(self, fig, name):
        output_dir = Path('benchmark_plots')
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def __init__(self, csv_path):
    # Read the CSV file
        self.df = pd.read_csv(csv_path)
        
        # Clean up kernel names - assuming they should be strings like "kernel1_warpReduce"
        self.df['Kernel_Name'] = self.df['Kernel_Name'].fillna('Unknown')
        self.df['Kernel_Name'] = self.df['Kernel_Name'].astype(str)
        
        # Convert Thread_Config to string
        self.df['Thread_Config'] = self.df['Thread_Config'].astype(str)
        
        # Setup style
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def plot_execution_times(self):
        print("\nShape of dataframe:", self.df.shape)
        print("\nColumns in dataframe:", self.df.columns.tolist())
        print("\nSample of data:")
        print(self.df.head())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        thread_configs = sorted(self.df['Thread_Config'].unique(), 
                            key=lambda x: int(x))  # Sort numerically
        kernels = sorted(self.df['Kernel_Name'].unique())
        
        # Remove 'Unknown' or 'nan' from kernels if present
        kernels = [k for k in kernels if k not in ['Unknown', 'nan']]
        
        # Create aggregated data for plotting
        plot_data = {}
        for kernel in kernels:
            plot_data[kernel] = []
            for config in thread_configs:
                mean_time = self.df[
                    (self.df['Kernel_Name'] == kernel) & 
                    (self.df['Thread_Config'] == config)
                ]['Execution_Time_ms'].mean()
                plot_data[kernel].append(mean_time)
        
        # Plot bars
        x = np.arange(len(thread_configs))
        width = 0.35 / len(kernels)
        
        for i, kernel in enumerate(kernels):
            if pd.isna(plot_data[kernel]).any():
                continue
            ax.bar(x + i*width, plot_data[kernel], width, 
                label=f"Kernel {kernel}", color=self.colors[i])
        
        ax.set_xlabel('Thread Configuration')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Average Kernel Execution Times Across Thread Configurations')
        ax.set_xticks(x + width*len(kernels)/2)
        ax.set_xticklabels(thread_configs, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        self.save_plot(fig, 'execution_times')

    def plot_power_metrics(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Power consumption
        sns.boxplot(data=self.df, x='Thread_Config', y='Avg_Power_mW', 
                hue='Kernel_Name', ax=ax1)
        ax1.set_title('Average Power Consumption')
        ax1.set_xlabel('Thread Configuration (X, Y)')
        ax1.set_ylabel('Power (mW)')
        ax1.tick_params(axis='x', rotation=45)

        # Energy consumption
        sns.boxplot(data=self.df, x='Thread_Config', y='Energy_mJ', 
                hue='Kernel_Name', ax=ax2)
        ax2.set_title('Energy Consumption')
        ax2.set_xlabel('Thread Configuration (X, Y)')
        ax2.set_ylabel('Energy (mJ)')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        self.save_plot(fig, 'power_metrics')

    def plot_efficiency_metrics(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate GFLOPS/Watt, handling zero/NaN values
        self.df['GFLOPS_per_Watt'] = np.where(
            (self.df['Energy_mJ'] > 0) & (self.df['Total_Threads'] > 0),
            (self.df['Total_Threads'] * 1e-9) / (self.df['Energy_mJ'] * 1e-3),
            0
        )
        
        # Group by Thread_Config and Kernel_Name to handle duplicates
        grouped_df = self.df.groupby(['Thread_Config', 'Kernel_Name'], as_index=False).agg({
            'GFLOPS_per_Watt': 'mean'
        })
        
        # Sort Thread_Config numerically
        grouped_df['Thread_Config'] = grouped_df['Thread_Config'].astype(float).astype(str)
        grouped_df = grouped_df.sort_values('Thread_Config')
        
        # Create the plot
        sns.barplot(data=grouped_df, 
                    x='Thread_Config', 
                    y='GFLOPS_per_Watt', 
                    hue='Kernel_Name',
                    ax=ax)
        
        ax.set_title('Computational Efficiency (GFLOPS/Watt)')
        ax.set_xlabel('Thread Configuration')
        ax.set_ylabel('GFLOPS/Watt')
        ax.tick_params(axis='x', rotation=45)
        
        # Adjust legend
        ax.legend(title='Kernel Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        self.save_plot(fig, 'efficiency_metrics')

    def plot_thread_scalability(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for kernel in self.df['Kernel_Name'].unique():
            kernel_data = self.df[self.df['Kernel_Name'] == kernel]
            ax.scatter(kernel_data['Total_Threads'], kernel_data['Execution_Time_ms'], 
                    label=kernel, alpha=0.7)
            ax.plot(kernel_data['Total_Threads'], kernel_data['Execution_Time_ms'])
            
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Total Number of Threads')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Thread Scalability Analysis')
        ax.legend()
        
        self.save_plot(fig, 'thread_scalability')

    def generate_summary_stats(self):
        summary = pd.DataFrame()
        
        for kernel in self.df['Kernel_Name'].unique():
            kernel_data = self.df[self.df['Kernel_Name'] == kernel]
            
            best_time_config = kernel_data.loc[kernel_data['Execution_Time_ms'].idxmin()]
            best_power_config = kernel_data.loc[kernel_data['Avg_Power_mW'].idxmin()]
            
            summary = pd.concat([summary, pd.DataFrame({
                'Kernel': [kernel],
                'Best_Time_Config': [best_time_config['Thread_Config']],
                'Best_Time_ms': [best_time_config['Execution_Time_ms']],
                'Best_Power_Config': [best_power_config['Thread_Config']],
                'Best_Power_mW': [best_power_config['Avg_Power_mW']]
            })])
            
        summary.to_csv('benchmark_plots/summary_stats.csv', index=False)
        return summary

def main():
    analyzer = BenchmarkAnalyzer('measurements.csv')
    
    # Generate all plots
    analyzer.plot_execution_times()
    analyzer.plot_power_metrics()
    analyzer.plot_efficiency_metrics()
    analyzer.plot_thread_scalability()
    
    # Generate summary statistics
    summary = analyzer.generate_summary_stats()
    print("\nSummary Statistics:")
    print(summary.to_string())

if __name__ == "__main__":
    main()