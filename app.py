"""
Solara Interactive Visualization for Cryptographic Parameter Optimization.

This module provides a modern, interactive web-based visualization using Solara.
Run with: solara run app.py
"""

import solara
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from model import CryptoOptimizationModel


# Reactive variables for model state
running = solara.reactive(False)
model_instance = solara.reactive(None)
current_step = solara.reactive(0)

# Model parameters
n_explorers = solara.reactive(20)
alpha = solara.reactive(0.7)
beta = solara.reactive(0.3)
topology = solara.reactive("ring")
n_steps = solara.reactive(100)


def reset_model():
    """Reset and create a new model instance."""
    running.value = False  # Stop running first
    current_step.value = 0
    model_instance.value = CryptoOptimizationModel(
        n_explorers=n_explorers.value,
        alpha=alpha.value,
        beta=beta.value,
        communication_topology=topology.value,
        seed=42
    )


def step_model():
    """Execute one step of the model."""
    if model_instance.value is not None:
        model_instance.value.step()
        current_step.value += 1
        
        # Debug: Print some agent states every 10 steps
        if current_step.value % 10 == 0:
            agent = model_instance.value.schedule.agents[0]
            print(f"\n=== Step {current_step.value} ===")
            print(f"Agent 0 params: n={agent.current_params['n']}, q={agent.current_params['q']:.1f}, σ={agent.current_params['sigma']:.3f}")
            print(f"Agent 0 velocity: q={agent.velocity['q']:.2f}, σ={agent.velocity['sigma']:.3f}")
            print(f"Agent 0 fitness: {agent.fitness_personal:.4f}")
        
        # Auto-stop when reaching max steps
        if current_step.value >= n_steps.value:
            running.value = False


@solara.component
def ControlPanel():
    """Compact control panel always visible at the top."""
    
    # Check if we can modify parameters (only when model is None or finished)
    can_modify = model_instance.value is None or (not running.value and current_step.value >= n_steps.value)
    
    with solara.Card(elevation=3, style="background: white; padding: 20px; margin-bottom: 20px;"):
        with solara.Row(style="gap: 20px; align-items: center;"):
            # Left: Parameters
            with solara.Column(style="flex: 2;"):
                solara.Markdown("### ⚙️ Configuration", style="margin: 0 0 10px 0;")
                with solara.Row(style="gap: 15px;"):
                    with solara.Column(style="flex: 1;"):
                        solara.SliderInt("Agents", value=n_explorers, min=5, max=50, step=5, disabled=not can_modify)
                        solara.SliderFloat("α (Security)", value=alpha, min=0.0, max=1.0, step=0.1, disabled=not can_modify)
                        solara.Select("Topology", value=topology, values=["ring", "random", "all"], disabled=not can_modify)
                    with solara.Column(style="flex: 1;"):
                        solara.SliderFloat("β (Performance)", value=beta, min=0.0, max=1.0, step=0.1, disabled=not can_modify)
                        solara.SliderInt("Steps", value=n_steps, min=10, max=500, step=10, disabled=not can_modify)
            
            # Center: Controls
            with solara.Column(style="flex: 1; gap: 10px;"):
                solara.Markdown("### 🎮 Controls", style="margin: 0 0 10px 0;")
                with solara.Row(style="gap: 10px;"):
                    solara.Button("🔄 Reset", on_click=reset_model, color="primary", disabled=running.value, style="flex: 1;")
                    solara.Button("➡️ Step", on_click=step_model, disabled=model_instance.value is None or running.value, style="flex: 1;")
                
                if not running.value:
                    solara.Button(
                        "▶️ RUN",
                        on_click=lambda: running.set(True),
                        color="success",
                        disabled=model_instance.value is None or current_step.value >= n_steps.value,
                        style="width: 100%; font-weight: bold; font-size: 1.1em;"
                    )
                else:
                    solara.Button(
                        "⏸️ PAUSE",
                        on_click=lambda: running.set(False),
                        color="warning",
                        style="width: 100%; font-weight: bold; font-size: 1.1em;"
                    )
            
            # Right: Status
            with solara.Column(style="flex: 1;"):
                solara.Markdown("### 📊 Status", style="margin: 0 0 10px 0;")
                if model_instance.value is not None:
                    progress = (current_step.value / n_steps.value) * 100
                    solara.ProgressLinear(value=progress, color="primary")
                    solara.Markdown(f"**Step {current_step.value} / {n_steps.value}** ({progress:.0f}%)", style="text-align: center; margin-top: 5px;")
                    
                    if running.value:
                        solara.Markdown("▶️ **Running...**", style="color: #4CAF50; text-align: center; font-weight: bold;")
                    elif current_step.value >= n_steps.value:
                        solara.Markdown("✅ **Completed! Click Reset for new run**", style="color: #2196F3; text-align: center; font-weight: bold;")
                    else:
                        solara.Markdown("⏸️ **Paused**", style="color: #FF9800; text-align: center;")
                else:
                    solara.Markdown("**Click Reset to start**", style="text-align: center; color: #666;")


@solara.component
def BestParameters():
    """Component displaying the best parameters found."""
    
    with solara.Card("🔒 Best Parameters Found", elevation=4, style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;"):
        if model_instance.value is None or model_instance.value.global_best_params is None:
            solara.Markdown("_No optimization results yet. Click Reset to start._", style="color: white;")
        else:
            model = model_instance.value
            params = model.global_best_params
            security = model.get_security_level(params)
            cost = model.get_performance_cost(params)
            
            with solara.Column(style="gap: 8px;"):
                # Parameters section
                with solara.Card(elevation=2, style="background: rgba(255,255,255,0.15); padding: 12px;"):
                    solara.Markdown("**📐 Parameters**", style="color: white; font-size: 1.1em;")
                    solara.Text(f"n: {params['n']}", style="color: white; font-size: 1.1em; font-weight: bold;")
                    solara.Text(f"q: {params['q']}", style="color: white; font-size: 1.1em; font-weight: bold;")
                    solara.Text(f"σ: {params['sigma']:.3f}", style="color: white; font-size: 1.1em; font-weight: bold;")
                
                # Metrics section
                with solara.Card(elevation=2, style="background: rgba(255,255,255,0.15); padding: 12px;"):
                    solara.Markdown("**📊 Metrics**", style="color: white; font-size: 1.1em;")
                    solara.Text(f"🔐 Security: {security:.1f} bits", style="color: #90EE90; font-size: 1.1em; font-weight: bold;")
                    solara.Text(f"⚡ Cost: {cost:.2f}", style="color: #FFD700; font-size: 1.1em; font-weight: bold;")
                    solara.Text(f"🎯 Fitness: {model.global_best_fitness:.2f}", style="color: #87CEEB; font-size: 1.1em; font-weight: bold;")
                    solara.Text(f"📈 Convergence: {model.compute_convergence_rate()*100:.1f}%", style="color: #FFA07A; font-size: 1.1em; font-weight: bold;")


@solara.component
def FitnessChart():
    """Component showing fitness evolution over time."""
    
    if model_instance.value is None:
        with solara.Card("Fitness Evolution", elevation=2):
            solara.Markdown("_Initialize the model to see charts_")
        return
    
    model_data = model_instance.value.datacollector.get_model_vars_dataframe()
    
    if len(model_data) == 0:
        with solara.Card("Fitness Evolution", elevation=2):
            solara.Markdown("_Run the model to see fitness evolution_")
        return
    
    # Create figure - will update when current_step changes
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(model_data.index, model_data['Global_Best_Fitness'], 
            label='Global Best', linewidth=2, color='darkblue')
    ax.plot(model_data.index, model_data['Average_Fitness'], 
            label='Average', linewidth=2, color='orange', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Evolution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    with solara.Card("Fitness Evolution", elevation=2):
        # Update only when step changes
        solara.FigureMatplotlib(fig, dependencies=[current_step.value])
    
    plt.close(fig)


@solara.component
def DiversityChart():
    """Component showing population diversity over time."""
    
    # Force re-render when current_step changes
    _ = current_step.value
    
    if model_instance.value is None:
        return
    
    model_data = model_instance.value.datacollector.get_model_vars_dataframe()
    
    if len(model_data) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(model_data.index, model_data['Diversity'], 
            linewidth=2, color='green')
    ax.set_xlabel('Step')
    ax.set_ylabel('Diversity (std of n)')
    ax.set_title('Population Diversity')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    with solara.Card("Population Diversity", elevation=2):
        solara.FigureMatplotlib(fig, dependencies=[current_step.value])
    
    plt.close(fig)


@solara.component
def ConvergenceChart():
    """Component showing convergence rate over time."""
    
    # Force re-render when current_step changes
    _ = current_step.value
    
    if model_instance.value is None:
        return
    
    model_data = model_instance.value.datacollector.get_model_vars_dataframe()
    
    if len(model_data) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(model_data.index, model_data['Convergence_Rate'] * 100, 
            linewidth=2, color='purple')
    ax.set_xlabel('Step')
    ax.set_ylabel('Convergence (%)')
    ax.set_title('Convergence Rate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    
    with solara.Card("Convergence Rate", elevation=2):
        solara.FigureMatplotlib(fig, dependencies=[current_step.value])
    
    plt.close(fig)


@solara.component
def ParametersEvolution():
    """Component showing parameter evolution over time."""
    
    # Force re-render when current_step changes
    _ = current_step.value
    
    if model_instance.value is None:
        return
    
    model_data = model_instance.value.datacollector.get_model_vars_dataframe()
    
    if len(model_data) == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Parameter n
    axes[0].plot(model_data.index, model_data['Global_Best_N'], 
                linewidth=2, color='red')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('n (dimension)')
    axes[0].set_title('Best Lattice Dimension (n)')
    axes[0].grid(True, alpha=0.3)
    
    # Parameter q
    axes[1].plot(model_data.index, model_data['Global_Best_Q'], 
                linewidth=2, color='blue')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('q (modulus)')
    axes[1].set_title('Best Modulus (q)')
    axes[1].grid(True, alpha=0.3)
    
    # Parameter sigma
    axes[2].plot(model_data.index, model_data['Global_Best_Sigma'], 
                linewidth=2, color='magenta')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('σ (noise)')
    axes[2].set_title('Best Noise Standard Deviation (σ)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    with solara.Card("Parameters Evolution", elevation=2):
        solara.FigureMatplotlib(fig, dependencies=[current_step.value])
    
    plt.close(fig)


@solara.component
def ParameterSpace2D():
    """Component showing agents' positions and trajectories in 2D parameter space (q vs sigma)."""
    
    # Force re-render when current_step changes for animation
    _ = current_step.value
    
    if model_instance.value is None:
        return
    
    agent_data = model_instance.value.datacollector.get_agent_vars_dataframe()
    
    if len(agent_data) == 0:
        return
    
    # Get latest step data
    latest_step = agent_data.index.get_level_values('Step').max()
    latest_data = agent_data.xs(latest_step, level='Step')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw trajectories for all agents
    agent_ids = agent_data.index.get_level_values('AgentID').unique()
    for agent_id in agent_ids:
        agent_trajectory = agent_data.xs(agent_id, level='AgentID')
        ax.plot(agent_trajectory['Q'], agent_trajectory['Sigma'], 
               alpha=0.3, linewidth=1, color='gray')
    
    # Plot current positions of all agents
    scatter = ax.scatter(latest_data['Q'], latest_data['Sigma'], 
                        c=latest_data['Fitness'], cmap='viridis', 
                        s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                        label='Current Positions')
    
    # Plot global best with a star marker
    if model_instance.value.global_best_params:
        best = model_instance.value.global_best_params
        ax.scatter(best['q'], best['sigma'], 
                  marker='*', s=250, c='red', 
                  edgecolors='darkred', linewidth=2,
                  label='Global Best', zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fitness', rotation=270, labelpad=20)
    
    # Fix axes limits to show full parameter space (prevents agents from "disappearing")
    ax.set_xlim(2000, 8500)  # q bounds with margin
    ax.set_ylim(1.8, 5.2)    # sigma bounds with margin
    
    ax.set_xlabel('q (Modulus)', fontsize=12, fontweight='bold')
    ax.set_ylabel('σ (Noise Std)', fontsize=12, fontweight='bold')
    ax.set_title(f'Agent Trajectories in Parameter Space (Step {latest_step})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    with solara.Card("🗺️ Parameter Space Exploration (q vs σ)", elevation=2):
        solara.FigureMatplotlib(fig, dependencies=[current_step.value])
    
    plt.close(fig)


@solara.component
def ParameterSpace3D():
    """Component showing agents' positions and trajectories in 3D parameter space."""
    
    # Force re-render when current_step changes for animation
    _ = current_step.value
    
    if model_instance.value is None:
        return
    
    agent_data = model_instance.value.datacollector.get_agent_vars_dataframe()
    
    if len(agent_data) == 0:
        return
    
    # Get latest step data
    latest_step = agent_data.index.get_level_values('Step').max()
    latest_data = agent_data.xs(latest_step, level='Step')
    
    # Create figure with 3D projection
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw trajectories for all agents
    agent_ids = agent_data.index.get_level_values('AgentID').unique()
    for agent_id in agent_ids:
        agent_trajectory = agent_data.xs(agent_id, level='AgentID')
        ax.plot(agent_trajectory['N'], agent_trajectory['Q'], agent_trajectory['Sigma'],
               alpha=0.3, linewidth=1, color='gray')
    
    # Plot current positions of all agents
    scatter = ax.scatter(latest_data['N'], latest_data['Q'], latest_data['Sigma'],
                        c=latest_data['Fitness'], cmap='plasma', 
                        s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                        label='Current Positions')
    
    # Plot global best with star marker
    if model_instance.value.global_best_params:
        best = model_instance.value.global_best_params
        ax.scatter(best['n'], best['q'], best['sigma'],
                  marker='*', s=250, c='red', 
                  edgecolors='darkred', linewidth=2,
                  label='Global Best', zorder=5)
    
    # Labels and title
    ax.set_xlabel('n (Dimension)', fontsize=11, fontweight='bold')
    ax.set_ylabel('q (Modulus)', fontsize=11, fontweight='bold')
    ax.set_zlabel('σ (Noise)', fontsize=11, fontweight='bold')
    ax.set_title(f'3D Agent Trajectories (Step {latest_step})', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Fix axes limits to show full parameter space (prevents agents from "disappearing")
    ax.set_xlim(200, 2100)   # n bounds with margin
    ax.set_ylim(2000, 8500)  # q bounds with margin
    ax.set_zlim(1.8, 5.2)    # sigma bounds with margin
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Fitness', rotation=270, labelpad=15)
    
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    with solara.Card("🎯 3D Parameter Space Exploration", elevation=2):
        solara.FigureMatplotlib(fig, dependencies=[current_step.value])
    
    plt.close(fig)


@solara.component
def Page():
    """Main application page."""
    
    # Create a separate reactive variable for auto-stepping
    tick = solara.use_reactive(0)
    
    # Auto-step effect that doesn't cause infinite loop
    def auto_step():
        if running.value and model_instance.value is not None and current_step.value < n_steps.value:
            # Schedule next step
            import threading
            import time
            def delayed_step():
                time.sleep(0.2)  # Slower update (200ms) to reduce visual jumps
                if running.value and current_step.value < n_steps.value:
                    step_model()
                    tick.value += 1
            threading.Thread(target=delayed_step, daemon=True).start()
    
    solara.use_effect(auto_step, [tick.value, running.value])
    
    with solara.Column(style="min-height: 100vh; background: #f5f5f5; padding: 0;"):
        # Header
        with solara.Card(elevation=2, style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; margin: 0; border-radius: 0;"):
            solara.Markdown("# 🔐 Cryptographic Parameter Optimization", style="margin: 0; color: white;")
            solara.Markdown("Multi-agent system optimizing LWE parameters using PSO", style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9); font-size: 0.95em;")
        
        # Main content with padding
        with solara.Column(style="padding: 20px; gap: 20px;"):
            # Control Panel (always visible at top)
            ControlPanel()
            
            if model_instance.value is not None:
                # Force update when current_step changes
                _ = current_step.value
                
                # Best Parameters Summary (compact)
                with solara.Card(elevation=2, style="background: white; padding: 15px;"):
                    with solara.Row(style="gap: 30px; align-items: center;"):
                        solara.Markdown("### 🏆 Best Parameters Found", style="margin: 0;")
                        
                        if model_instance.value.global_best_params:
                            params = model_instance.value.global_best_params
                            security = model_instance.value.get_security_level(params)
                            
                            with solara.Row(style="gap: 20px; flex: 1;"):
                                solara.Markdown(f"**n:** {params['n']}", style="margin: 0;")
                                solara.Markdown(f"**q:** {params['q']}", style="margin: 0;")
                                solara.Markdown(f"**σ:** {params['sigma']:.3f}", style="margin: 0;")
                                solara.Markdown(f"🔐 **Security:** {security:.0f}", style="margin: 0; color: #4CAF50; font-size: 1.1em;")
                                solara.Markdown(f"🎯 **Fitness:** {model_instance.value.global_best_fitness:.2f}", style="margin: 0; color: #2196F3; font-size: 1.1em;")
                        
                        # Info about PSO exploration
                        with solara.Row(style="margin-top: 10px;"):
                            solara.Markdown("_ℹ️ Note: 'n' changes discretely (256→512→1024→2048). PSO explores 'q' and 'σ' continuously. Security is an optimization metric (not realistic bits)._", style="font-size: 0.85em; color: #666; font-style: italic;")
                
                # Main visualizations
                with solara.Column(style="gap: 20px;"):
                    # Top row: Parameter space animations (MAIN FOCUS)
                    solara.Markdown("## 🎬 Dynamic Agent Visualization", style="margin: 10px 0;")
                    with solara.Row(style="gap: 20px;"):
                        with solara.Column(style="flex: 1;"):
                            ParameterSpace2D()
                        with solara.Column(style="flex: 1;"):
                            ParameterSpace3D()
                    
                    # Middle row: Fitness and convergence
                    solara.Markdown("## 📈 Optimization Metrics", style="margin: 20px 0 10px 0;")
                    with solara.Row(style="gap: 20px;"):
                        with solara.Column(style="flex: 2;"):
                            FitnessChart()
                        with solara.Column(style="flex: 1;"):
                            ConvergenceChart()
                    
                    # Bottom row: Diversity and parameters evolution
                    with solara.Row(style="gap: 20px;"):
                        with solara.Column(style="flex: 1;"):
                            DiversityChart()
                        with solara.Column(style="flex: 2;"):
                            ParametersEvolution()
            else:
                # Welcome screen
                with solara.Card(elevation=4, style="background: white; padding: 60px; text-align: left; margin-top: 40px;"):
                    solara.Markdown("""
                    # 👋 Welcome!
                    
                    ### To get started:
                    
                    1. **Adjust parameters** in the panel above
                    2. **Click Reset** to initialize the model
                    3. **Click RUN** to start optimization
                    
                    You'll see agents moving in real-time through parameter space! 🚀
                    """, style="color: #333; font-size: 1.1em;")
