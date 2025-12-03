import pandas as pd
import numpy as np  # Needed for simulating the data breakdown
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --- 1. Original DataFrame ---
data = {
    'rmse': [40, 90, 95, 130, 170, 205, 215, 220, 305, 320, 440, 470, 475, 480],
    'composite_loss': [150000, 0, 0, 1350000, 150000, 450000, 5000, 800000, 480000, 140000, 480000, 1225000, 225000, 1500000]
}
df_composite_loss = pd.DataFrame(data)

# --- 2. Simulate the Breakdown of Composite Loss ---
# IMPORTANT: This section is for demonstration.
# In your actual use case, you should replace these simulated columns
# with your real, calculated curtailment and shortfall data.

# We create a random split for each row.
np.random.seed(42) # for reproducible results
split_ratio = np.random.rand(len(df_composite_loss))

# Calculate the two components based on the random split
df_composite_loss['shortfall_cost'] = df_composite_loss['composite_loss'] * split_ratio
df_composite_loss['curtailment_cost'] = df_composite_loss['composite_loss'] * (1 - split_ratio)

# Sort by RMSE for a more organized plot
df_plot = df_composite_loss.sort_values('rmse').reset_index()


# --- 3. Generate the Stacked Bar Chart ---
print("Generating Plot 2: Composite Loss Breakdown vs. RMSE...")
if not df_plot.empty:
    # Set style and figure size
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig2, ax2 = plt.subplots(figsize=(16,8))

    # --- Plot the two components as stacked bars ---
    # Plot the BASE of the bars (curtailment cost)
    sns.barplot(x=df_plot.index, y='curtailment_cost', data=df_plot,
                color='darkcyan', label='Curtailment Cost', ax=ax2)

    # Plot the TOP of the bars (shortfall cost), starting from where the first bar ended
    sns.barplot(x=df_plot.index, y='shortfall_cost', data=df_plot,
                color='coral', label='Shortfall Cost',
                bottom=df_plot['curtailment_cost'], ax=ax2)

    # --- Styling and Labels (using your preferred large, bold fonts) ---
    ax2.set_title('Composite Loss Breakdown vs. Forecast Inaccuracy', fontsize=24, fontweight='bold')
    ax2.set_xlabel('Forecast RMSE (MW)', fontsize=24, fontweight='bold')
    ax2.set_ylabel('Component Loss ($)', fontsize=24, fontweight='bold')

    # Set the x-axis ticks to be the actual RMSE values
    ax2.set_xticks(df_plot.index)
    ax2.set_xticklabels(df_plot['rmse'], rotation=45, ha='right')

    # Format the y-axis ticks as currency
    ax2.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax2.tick_params(axis='both', which='major', labelsize=18)

    # --- Add and Style the Legend ---
    legend = ax2.legend()
    legend.get_title().set_fontweight('bold')
    plt.setp(legend.get_texts(), fontsize='16')

    # Final layout adjustments and save
    plt.tight_layout()
    plt.savefig('plot2_composite_loss_breakdown.png', dpi=300)
    plt.show()