import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --- 1. Recreate the DataFrame from the plot data ---
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

# --- 2. Plotting ---
print("Generating Plot with Zoomed Inset and a FULL Detailed Legend...")
if not df_models.empty:
    
    # --- Set a clean, bold style to match your image ---
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.linewidth': 2.5,
        'xtick.major.size': 8,
        'xtick.major.width': 2.5,
        'ytick.major.size': 8,
        'ytick.major.width': 2.5,
    })

    fig, ax = plt.subplots(figsize=(16, 9))

    # --- Define custom markers and palette ---
    model_markers = ["o", "X", "s", "P", "D", "D", "^", "s", "v"]
    palette = sns.color_palette("viridis", n_colors=len(df_models))

    # --- MAIN PLOT ---
    # We let Seaborn create the default legend so we can capture its contents
    sns.scatterplot(
        x='RMSE', y='Cost', data=df_models,
        hue='Model Type', style='Model Type',
        markers=model_markers, palette=palette,
        s=600, ax=ax, edgecolor='black', linewidth=2
    )
    
    # --- THIS IS THE CRITICAL FIX ---
    # 1. Get the handles and labels from the legend Seaborn just created.
    handles, labels = ax.get_legend_handles_labels()
    # 2. Now, immediately remove the temporary legend that Seaborn placed inside the plot.
    ax.get_legend().remove()
    # ---------------------------------
    
    # --- INSET PLOT (The Magnifying Glass) ---
    ax_inset = ax.inset_axes([0.45, 0.4, 0.4, 0.5])
    
    sns.scatterplot(
        x='RMSE', y='Cost', data=df_models,
        hue='Model Type', style='Model Type',
        markers=model_markers, palette=palette,
        s=600, ax=ax_inset, edgecolor='black', linewidth=2,
        legend=False # No legend needed on the inset
    )
    
    ax_inset.set_xlim(215, 280)
    ax_inset.set_ylim(1500, 6500)
    
    ax_inset.tick_params(axis='both', which='major', labelsize=14, width=2.5)
    ax_inset.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    ax_inset.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    for spine in ax_inset.spines.values():
        spine.set_linewidth(2.5)

    ax.indicate_inset_zoom(ax_inset, edgecolor="black", alpha=1, lw=2.5)

    # --- STYLING THE MAIN PLOT ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_title('Performance of Models Trained with Different Objectives', fontsize=24, loc='left')
    ax.set_xlabel('Resulting Forecast RMSE (MW)', fontsize=24)
    ax.set_ylabel('Resulting System Cost ($)', fontsize=24)
    ax.set_xlim(150, 1150)
    ax.set_ylim(-10000, 145000)

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{format(int(x), ",")}'))

    # --- CREATE THE FINAL LEGEND YOU WANT ---
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
    legend.get_frame().set_linewidth(2.5)
    legend.get_title().set_fontweight('bold')

    # Use tight_layout with a rect parameter to make space for the external legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('performance_plot_final_version.png', dpi=300)
    plt.show()