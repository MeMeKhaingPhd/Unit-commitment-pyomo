import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'rmse': [40, 90, 95, 130, 170, 205, 215, 220, 305, 320, 440, 470, 475, 480],
    'composite_loss': [150000, 0, 0, 1350000, 150000, 450000, 5000, 800000, 480000, 140000, 480000, 1225000, 225000, 1500000]
}
df_composite_loss = pd.DataFrame(data)

print("Generating Plot 1: Composite Loss vs. RMSE...")
if not df_composite_loss.empty:
    # Increased figsize to better fit the large fonts
    fig1, ax1 = plt.subplots(figsize=(16, 8))

    # Set a light background style similar to the original image
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    # Plot the data
    sns.regplot(
        x='rmse',
        y='composite_loss',
        data=df_composite_loss,
        ax=ax1,
        color='teal',
        # Increased marker size 
        scatter_kws={'alpha': 0.9, 's': 150}
    )

    # Increased font size and made bold 
    ax1.set_title('Composite Loss vs. Forecast Inaccuracy', fontsize=24, fontweight='bold')
    ax1.set_xlabel('Forecast RMSE (MW)', fontsize=24, fontweight='bold')
    ax1.set_ylabel('Composite Loss ($)', fontsize=24, fontweight='bold')

    #  Increased tick label size for readability 
    ax1.tick_params(axis='both', which='major', labelsize=22)

    # Keep original y-axis formatting and zero line
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${format(int(x), ",")}'))
    ax1.axhline(0, color='black', lw=1.5, linestyle='-')

    # Apply layout and save the figure
    plt.tight_layout()
    plt.savefig('plot1_composite_loss_large_font.png', dpi=300)
    plt.show()