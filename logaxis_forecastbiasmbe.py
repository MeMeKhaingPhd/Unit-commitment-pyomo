import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --- Create a sample DataFrame that includes MBE values ---
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
    'Cost': [6000, 5000, 4500, 3500, 4000, 135000, 3000, 2000, 4000],
    'MBE': [-120, -100, -130, -80, -75, -780, -60, -50, -110]
}
df_all_models = pd.DataFrame(data)
# --------------------------------------------------------------------------


#  Plot 4: Bar chart comparing Bias with a Symmetrical Log Axis
print("Generating Plot 4: Resulting Forecast Bias (Symlog Scale)...")
if not df_all_models.empty:
    sns.set_style("whitegrid")

    fig4, ax4 = plt.subplots(figsize=(16, 9))

    # Sort by 'Cost' to maintain the same model order as the economic ranking plot
    df_sorted_for_bias_plot = df_all_models.sort_values('Cost', ascending=True)
    
    sns.barplot(
        x='MBE',
        y='Model Type',
        data=df_sorted_for_bias_plot,
        ax=ax4,
        palette='plasma'
    )
    
    # --- NEW: Set the x-axis to a symmetrical logarithmic scale ---
    # This is the correct way to handle log scales with negative values.
    # 'linthresh' defines the range around zero that remains linear.
    ax4.set_xscale('symlog', linthresh=50)
    
    # --- MODIFIED: Updated titles and labels to reflect the new scale ---
    ax4.set_title('Resulting Forecast Bias (MBE) of Models', fontsize=24, fontweight='bold')
    ax4.set_xlabel('Mean Bias Error (MW) [Symlog Scale]', fontsize=24, fontweight='bold')
    ax4.set_ylabel('') # Keep y-label empty for a clean look

    ax4.tick_params(axis='both', which='major', labelsize=20)
    
    # Ensure the formatter handles negative numbers correctly
    ax4.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    # Ensure layout is tight and save the figure
    plt.tight_layout()
    plt.savefig('plot4_bias_comparison_symlog_scale.png', dpi=300)
    plt.show()