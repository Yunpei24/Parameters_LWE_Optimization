"""
Main script to run the Collaborative Cryptographic Parameter Optimization simulation.

This script demonstrates different usage patterns:
1. Basic simulation run
2. Comparison of different topologies
3. Sensitivity analysis
"""

import matplotlib.pyplot as plt
import pandas as pd
from model import CryptoOptimizationModel


def run_single_simulation(n_steps=100, topology="ring", n_explorers=20):
    """
    Run a single simulation and plot results.
    
    Args:
        n_steps: Number of simulation steps
        topology: Communication topology ("ring", "all", "random")
        n_explorers: Number of explorer agents
    """
    print(f"\n{'='*60}")
    print(f"Running simulation with {topology} topology...")
    print(f"Number of explorers: {n_explorers}")
    print(f"Number of steps: {n_steps}")
    print(f"{'='*60}\n")
    
    # Create and run model
    model = CryptoOptimizationModel(
        n_explorers=n_explorers,
        alpha=0.7,
        beta=0.3,
        communication_topology=topology,
        seed=42
    )
    
    model.run_model(n_steps)
    
    # Get results
    results = model.get_results_summary()
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best Parameters Found:")
    print(f"  n (dimension):     {results['best_params']['n']}")
    print(f"  q (modulus):       {results['best_params']['q']}")
    print(f"  σ (noise std):     {results['best_params']['sigma']:.3f}")
    print(f"\nPerformance Metrics:")
    print(f"  Security Level:    {results['security_bits']:.1f} bits")
    print(f"  Performance Cost:  {results['performance_cost']:.2f}")
    print(f"  Best Fitness:      {results['best_fitness']:.2f}")
    print(f"  Avg Fitness:       {results['avg_fitness']:.2f}")
    print(f"  Convergence Rate:  {results['convergence']*100:.1f}%")
    print("="*60 + "\n")
    
    # Plot results
    plot_optimization_results(model)
    
    return model, results


def plot_optimization_results(model):
    """
    Create comprehensive plots of optimization results.
    
    Args:
        model: Trained CryptoOptimizationModel instance
    """
    # Get data
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Collaborative Cryptographic Parameter Optimization Results', 
                 fontsize=16, fontweight='bold')
    
    # 1. Fitness Evolution
    ax = axes[0, 0]
    ax.plot(model_data.index, model_data['Global_Best_Fitness'], 
            label='Global Best', linewidth=2, color='darkblue')
    ax.plot(model_data.index, model_data['Average_Fitness'], 
            label='Average', linewidth=2, color='orange', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Evolution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Diversity
    ax = axes[0, 1]
    ax.plot(model_data.index, model_data['Diversity'], 
            linewidth=2, color='green')
    ax.set_xlabel('Step')
    ax.set_ylabel('Diversity (std of n)')
    ax.set_title('Population Diversity')
    ax.grid(True, alpha=0.3)
    
    # 3. Convergence Rate
    ax = axes[0, 2]
    ax.plot(model_data.index, model_data['Convergence_Rate'] * 100, 
            linewidth=2, color='purple')
    ax.set_xlabel('Step')
    ax.set_ylabel('Convergence (%)')
    ax.set_title('Convergence Rate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 4. Parameter n Evolution
    ax = axes[1, 0]
    ax.plot(model_data.index, model_data['Global_Best_N'], 
            linewidth=2, color='red')
    ax.set_xlabel('Step')
    ax.set_ylabel('n (dimension)')
    ax.set_title('Best Lattice Dimension (n)')
    ax.grid(True, alpha=0.3)
    
    # 5. Parameter q Evolution
    ax = axes[1, 1]
    ax.plot(model_data.index, model_data['Global_Best_Q'], 
            linewidth=2, color='blue')
    ax.set_xlabel('Step')
    ax.set_ylabel('q (modulus)')
    ax.set_title('Best Modulus (q)')
    ax.grid(True, alpha=0.3)
    
    # 6. Parameter sigma Evolution
    ax = axes[1, 2]
    ax.plot(model_data.index, model_data['Global_Best_Sigma'], 
            linewidth=2, color='magenta')
    ax.set_xlabel('Step')
    ax.set_ylabel('σ (noise)')
    ax.set_title('Best Noise Standard Deviation (σ)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/optimization_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'optimization_results.png'")
    plt.show()


def compare_topologies(n_steps=100, n_runs=5):
    """
    Compare performance of different communication topologies.
    
    Args:
        n_steps: Number of steps per simulation
        n_runs: Number of runs per topology for averaging
    """
    print("\n" + "="*60)
    print("COMPARING COMMUNICATION TOPOLOGIES")
    print("="*60 + "\n")
    
    topologies = ["ring", "random", "all"]
    results_comparison = {top: [] for top in topologies}
    
    for topology in topologies:
        print(f"Testing {topology} topology...")
        
        for run in range(n_runs):
            model = CryptoOptimizationModel(
                n_explorers=20,
                alpha=0.7,
                beta=0.3,
                communication_topology=topology,
                seed=42 + run
            )
            
            model.run_model(n_steps)
            results = model.get_results_summary()
            results_comparison[topology].append(results['best_fitness'])
        
        avg_fitness = sum(results_comparison[topology]) / n_runs
        print(f"  Average best fitness: {avg_fitness:.2f}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = range(len(topologies))
    means = [sum(results_comparison[top]) / n_runs for top in topologies]
    
    bars = ax.bar(positions, means, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_xticks(positions)
    ax.set_xticklabels(topologies)
    ax.set_ylabel('Average Best Fitness')
    ax.set_title('Comparison of Communication Topologies')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./results/topology_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'topology_comparison.png'")
    plt.show()
    
    return results_comparison


def sensitivity_analysis():
    """
    Analyze sensitivity to alpha/beta parameters.
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS (Alpha/Beta)")
    print("="*60 + "\n")
    
    alpha_values = [0.3, 0.5, 0.7, 0.9]
    results_sensitivity = []
    
    for alpha in alpha_values:
        beta = 1 - alpha
        print(f"Testing α={alpha:.1f}, β={beta:.1f}...")
        
        model = CryptoOptimizationModel(
            n_explorers=20,
            alpha=alpha,
            beta=beta,
            communication_topology="ring",
            seed=42
        )
        
        model.run_model(50)
        results = model.get_results_summary()
        
        results_sensitivity.append({
            'alpha': alpha,
            'beta': beta,
            'best_fitness': results['best_fitness'],
            'security_bits': results['security_bits'],
            'performance_cost': results['performance_cost']
        })
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    alphas = [r['alpha'] for r in results_sensitivity]
    
    # Fitness
    axes[0].plot(alphas, [r['best_fitness'] for r in results_sensitivity], 
                 marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('α (Security Weight)')
    axes[0].set_ylabel('Best Fitness')
    axes[0].set_title('Fitness vs. Alpha')
    axes[0].grid(True, alpha=0.3)
    
    # Security
    axes[1].plot(alphas, [r['security_bits'] for r in results_sensitivity], 
                 marker='s', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('α (Security Weight)')
    axes[1].set_ylabel('Security (bits)')
    axes[1].set_title('Security Level vs. Alpha')
    axes[1].grid(True, alpha=0.3)
    
    # Cost
    axes[2].plot(alphas, [r['performance_cost'] for r in results_sensitivity], 
                 marker='^', linewidth=2, markersize=8, color='red')
    axes[2].set_xlabel('α (Security Weight)')
    axes[2].set_ylabel('Performance Cost')
    axes[2].set_title('Computational Cost vs. Alpha')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'sensitivity_analysis.png'")
    plt.show()
    
    return results_sensitivity


if __name__ == "__main__":
    # Run different analysis modes
    
    print("\n" + "#"*60)
    print("# COLLABORATIVE CRYPTOGRAPHIC OPTIMIZATION SIMULATION")
    print("#"*60)
    
    # 1. Single simulation with ring topology
    model, results = run_single_simulation(
        n_steps=200, 
        topology="random", 
        n_explorers=40
    )
    
    # 2. Compare topologies (uncomment to run)
    compare_topologies(n_steps=100, n_runs=5)
    
    # 3. Sensitivity analysis (uncomment to run)
    sensitivity_analysis()
    
    print("\n✓ Simulation completed successfully!\n")