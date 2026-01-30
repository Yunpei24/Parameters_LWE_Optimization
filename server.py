"""
Interactive visualization server for the Cryptographic Optimization Model.

This module creates a browser-based interactive visualization using Mesa's
visualization framework.
"""

import mesa
from model import CryptoOptimizationModel


def agent_portrayal(agent):
    """
    Define how agents are portrayed in the visualization.
    
    Args:
        agent: ExplorerAgent instance
    
    Returns:
        dict: Visualization properties for the agent
    """
    # Color based on fitness (normalized)
    # Higher fitness = more green, lower = more red
    max_fitness = 300  # Approximate max fitness
    normalized_fitness = min(agent.fitness_personal / max_fitness, 1.0)
    
    # Color gradient from red to yellow to green
    if normalized_fitness < 0.5:
        r = 255
        g = int(255 * (normalized_fitness * 2))
    else:
        r = int(255 * (1 - (normalized_fitness - 0.5) * 2))
        g = 255
    
    color = f"rgb({r}, {g}, 0)"
    
    # Size based on how close to personal best
    size = 8 + (normalized_fitness * 12)
    
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "Color": color,
        "r": size,
        "Layer": 0,
        "text": f"Agent {agent.unique_id}",
        "text_color": "white"
    }
    
    return portrayal


# Model parameters that can be adjusted in the interface
model_params = {
    "n_explorers": mesa.visualization.Slider(
        "Number of Explorers",
        value=20,
        min_value=5,
        max_value=50,
        step=5,
        description="Number of agents exploring parameter space"
    ),
    "alpha": mesa.visualization.Slider(
        "Security Weight (α)",
        value=0.7,
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        description="Weight for security in fitness function"
    ),
    "beta": mesa.visualization.Slider(
        "Performance Weight (β)",
        value=0.3,
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        description="Weight for performance cost in fitness function"
    ),
    "communication_topology": mesa.visualization.Choice(
        "Communication Topology",
        value="ring",
        choices=["ring", "random", "all"],
        description="How agents share information"
    )
}


# Chart: Global Best Fitness over time
fitness_chart = mesa.visualization.ChartModule(
    [
        {"Label": "Global_Best_Fitness", "Color": "#0066CC"},
        {"Label": "Average_Fitness", "Color": "#FF6600"}
    ],
    data_collector_name='datacollector',
    canvas_height=200,
    canvas_width=500
)

# Chart: Diversity over time
diversity_chart = mesa.visualization.ChartModule(
    [{"Label": "Diversity", "Color": "#00AA00"}],
    data_collector_name='datacollector',
    canvas_height=200,
    canvas_width=500
)

# Chart: Convergence rate
convergence_chart = mesa.visualization.ChartModule(
    [{"Label": "Convergence_Rate", "Color": "#AA00AA"}],
    data_collector_name='datacollector',
    canvas_height=200,
    canvas_width=500
)

# Chart: Best parameters evolution
parameters_chart = mesa.visualization.ChartModule(
    [
        {"Label": "Global_Best_N", "Color": "#FF0000"},
        {"Label": "Global_Best_Sigma", "Color": "#0000FF"}
    ],
    data_collector_name='datacollector',
    canvas_height=200,
    canvas_width=500
)


# Text display for current best parameters
def get_best_params_text(model):
    """Generate text display of best parameters."""
    if model.global_best_params is None:
        return "No optimal parameters found yet."
    
    params = model.global_best_params
    security = model.get_security_level(params)
    cost = model.get_performance_cost(params)
    
    text = f"""
    <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
        <h3 style='margin-top: 0;'>🔒 Best Parameters Found</h3>
        <table style='width: 100%;'>
            <tr>
                <td><strong>Dimension (n):</strong></td>
                <td>{params['n']}</td>
            </tr>
            <tr>
                <td><strong>Modulus (q):</strong></td>
                <td>{params['q']}</td>
            </tr>
            <tr>
                <td><strong>Noise (σ):</strong></td>
                <td>{params['sigma']:.3f}</td>
            </tr>
            <tr style='border-top: 2px solid #ccc;'>
                <td><strong>Security:</strong></td>
                <td>{security:.1f} bits</td>
            </tr>
            <tr>
                <td><strong>Cost:</strong></td>
                <td>{cost:.2f}</td>
            </tr>
            <tr>
                <td><strong>Fitness:</strong></td>
                <td>{model.global_best_fitness:.2f}</td>
            </tr>
        </table>
    </div>
    """
    return text


# Custom text element
class TextElement(mesa.visualization.TextElement):
    """Custom text element to display best parameters."""
    
    def render(self, model):
        return get_best_params_text(model)


best_params_display = TextElement()


# Create server
server = mesa.visualization.ModularServer(
    CryptoOptimizationModel,
    [
        best_params_display,
        fitness_chart,
        diversity_chart,
        convergence_chart,
        parameters_chart
    ],
    "Collaborative Cryptographic Parameter Optimization",
    model_params
)

server.port = 8521  # Default Mesa port


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Interactive Visualization Server")
    print("="*60)
    print("\nOpen your browser and navigate to:")
    print("http://127.0.0.1:8521/")
    print("\nPress Ctrl+C to stop the server.")
    print("="*60 + "\n")
    
    server.launch()