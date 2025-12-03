import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --- Create a sample DataFrame that matches the data from previous plots ---
# This makes the code runnable. You can replace this with your actual df_all_models.
data = {
    'Model Type': [
        'rho=0.75, kappa=0.25',
        'rho=0.80, kappa=0.20',
        'rho=0.85, kappa=0.15',
        'rho=0.90, kappa=0.10',
        'rho=0.95, kappa=0.05',
        'rho=1.00, kappa=0.00',
        'Standard MSE (L2)',
        'Standard MAE (L1)',
        'Standard Huber'
    ],
    'Cost': [6000, 5000, 4500, 3500, 4000, 135000, 3000, 2000, 4000]
}
df_all_models = pd.DataFrame(data)
# --------------------------------------------------------------------------


# Plot 3: Bar chart comparing Costs
print("Generating Plot 3: Economic Performance Ranking...")
if not df_all_models.empty:
    # --- MODIFICATION: Set a style similar to the image ---
    sns.set_style("whitegrid")
    
    # --- MODIFICATION: Increased figsize for better layout with large fonts ---
    fig3, ax3 = plt.subplots(figsize=(16, 9))
    
    # Sort the dataframe by 'Cost' so the best models (lowest cost) are at the top
    df_all_models_sorted = df_all_models.sort_values('Cost', ascending=True)
    
    sns.barplot(
        x='Cost',
        y='Model Type',
        data=df_all_models_sorted,
        ax=ax3,
        palette='plasma'
    )
    
    # --- MODIFICATION: Increased font size and made bold ---
    ax3.set_title('Economic Performance Ranking of Models', fontsize=24, fontweight='bold')
    ax3.set_xlabel('Total System Cost ($)', fontsize=24, fontweight='bold')
    ax3.set_ylabel('Training Objective', fontsize=24, fontweight='bold')
    
    # --- MODIFICATION: Increased tick label size for readability ---
    ax3.tick_params(axis='both', which='major', labelsize=20)

    # Keep the original x-axis currency formatting
    ax3.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    
    # Ensure layout is tight and save the figure
    plt.tight_layout()
    plt.savefig('plot3_cost_ranking_large_font.png', dpi=300)
    plt.show()