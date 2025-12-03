import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

data = {
    'rho': [0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
    'Cost': [6000, 5000, 4500, 3500, 4000, 135000]
}
df_asymmetric = pd.DataFrame(data)
# -------------------------------------------------------------


# Plot 5: System Cost vs. Rho value with a Log Axis
print("Generating Plot 5: Rho Sensitivity Analysis (Log Scale)...")
if not df_asymmetric.empty:
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    fig5, ax5 = plt.subplots(figsize=(16, 9))
    
    # Create the line plot with larger markers and a thicker line
    sns.lineplot(
        x='rho',
        y='Cost',
        data=df_asymmetric,
        ax=ax5,
        marker='o',
        color='navy',
        markersize=12,
        linewidth=3
    )
    
    # --- NEW: Set the y-axis to a logarithmic scale ---
    # This will highlight the variations in the lower-cost points.
    ax5.set_yscale('log')
    
    # --- MODIFIED: Updated titles and labels to reflect the log scale ---
    ax5.set_title('System Cost vs. Asymmetric Penalty (rho)', fontsize=24, fontweight='bold')
    ax5.set_xlabel('Rho value (Weight on Over-prediction)', fontsize=24, fontweight='bold')
    ax5.set_ylabel('Resulting System Cost ($) [Log Scale]', fontsize=24, fontweight='bold')
    
    ax5.tick_params(axis='both', which='major', labelsize=20)

    # The original y-axis formatter works correctly with a log scale
    ax5.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    
    # Ensure layout is tight and save the figure
    plt.tight_layout()
    plt.savefig('plot5_rho_sensitivity_log_scale.png', dpi=300)
    plt.show()