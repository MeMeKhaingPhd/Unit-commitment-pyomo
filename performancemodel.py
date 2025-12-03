import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --- 1. Recreate the DataFrame based on the new plot data ---
# Coordinates and model types are estimated from the provided image.
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
    'RMSE': [255, 260, 265, 270, 275, 1050, 240, 230, 220],
    'Cost': [6000, 5000, 4500, 3500, 4000, 135000, 3000, 2000, 4000]
}
df_models = pd.DataFrame(data)
# -------------------------------------------------------------


# --- 2. Plotting ---
print("Generating Plot: Model Performance...")
if not df_models.empty:
    # --- Increased figsize for better layout with large fonts ---
    fig, ax = plt.subplots(figsize=(16,9))

    # Set the background style to whitegrid with dashed lines
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    # --- Define custom markers and palette to match the image ---
    model_markers = ["o", "X", "s", "P", "D", "D", "^", "s", "v"]
    # Using 'viridis' palette as it closely matches the colors
    palette = sns.color_palette("viridis", n_colors=len(df_models))

    # Create the scatter plot
    sns.scatterplot(
        x='RMSE',
        y='Cost',
        data=df_models,
        hue='Model Type',
        style='Model Type',
        markers=model_markers,
        palette=palette,
        # --- Increased marker size ---
        s=500,
        ax=ax,
        edgecolor='black' # Add a thin border to markers for clarity
    )

    # --- 3. Set Titles and Labels with large, bold fonts ---
    ax.set_title('Performance of Models Trained with Different Objectives', fontsize=24, fontweight='bold')
    ax.set_xlabel('Resulting Forecast RMSE (MW)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Resulting System Cost ($)', fontsize=24, fontweight='bold')

    # --- 4. Format Axes and Ticks ---
    # Set axis limits to match the provided image
    ax.set_xlim(150, 1150)
    ax.set_ylim(-10000, 145000)

    # Format y-axis ticks as currency
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # --- Increase tick label font size ---
    ax.tick_params(axis='both', which='major', labelsize=22)

    # --- 5. Customize the Legend ---
    legend = ax.legend(
        title='Training Objective',
        bbox_to_anchor=(1.02, 1), # Position legend outside the plot area
        loc='upper left',
        borderaxespad=0.,
        # --- Set legend font sizes ---
        fontsize=20,
        title_fontsize=22
    )
    # Make legend title bold
    legend.get_title().set_fontweight('bold')
    # Add a frame to the legend
    legend.get_frame().set_edgecolor('grey')
    legend.get_frame().set_linewidth(1.5)

    # --- 6. Final Touches ---
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for the legend
    plt.savefig('model_performance_plot_large.png', dpi=300)
    plt.show()