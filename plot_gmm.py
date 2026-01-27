import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# --- Configuration for Publication-Quality Figure ---
sns.set_theme(style="white", font_scale=1.2)
plt.rcParams['font.family'] = 'serif' # Use serif font for paper
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def plot_gmm_illustration(output_path="gmm_learned_density.svg"):
    # --- 1. Define GMM Parameters representing your scenario ---
    # The residual 'r'
    x = np.linspace(-1, 1, 500)

    mu1, sigma1, weight1 = 0.0, 0.1, 0.5
    mu2, sigma2, weight2 = 0.7, 0.3, 0.3
    mu3, sigma3, weight3 = -0.5, 0.2, 0.5

    # Calculate individual weighted PDFs
    pdf1 = weight1 * stats.norm.pdf(x, mu1, sigma1)
    pdf2 = weight2 * stats.norm.pdf(x, mu2, sigma2)
    pdf3 = weight3 * stats.norm.pdf(x, mu3, sigma3)
    
    # Calculate Total Mixture PDF (This is the "Learned Noise Density")
    pdf_total = pdf1 + pdf2 + pdf3

    # --- 2. Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    # # Plot underlying components (dashed, lighter colors)
    # ax.plot(x, pdf1, color='green', linestyle='--', linewidth=2, alpha=0.6)
    # ax.fill_between(x, pdf1, color='green', alpha=0.1)

    # ax.plot(x, pdf2, color='red', linestyle='--', linewidth=2, alpha=0.6)
    # ax.fill_between(x, pdf2, color='red', alpha=0.1)

    # ax.plot(x, pdf3, color='orange', linestyle='--', linewidth=2, alpha=0.6)
    # ax.fill_between(x, pdf3, color='orange', alpha=0.1)
    # # Plot Total Density (Solid, prominent color)
    ax.plot(x, pdf_total, color='#E15759', linewidth=13, linestyle='--')
    # erase x-axis line and y-axis line
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # --- 3. Annotations and Styling ---
    # Main Title using your chosen terminology
    # ax.set_title("Illustration: Learned Noise Density (GMM)", fontsize=16, fontweight='bold', pad=20)
    
    # ax.set_xlabel("Residual Value", fontsize=50)
    # ax.set_ylabel("PDF", fontsize=50)
    ax.set_yticks([]) # Hide Y-ticks as exact density values don't matter for concept
    ax.set_xticks([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(bottom=0)

    # Add semantic annotations pointing to the peaks
    # ax.annotate('High Density\n(Normal Data)', xy=(mu1, pdf_total.max()), xytext=(-3, pdf_total.max()*0.8),
    #             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
    #             fontsize=12, ha='center')

    # ax.annotate('Low Density Cluster\n(Outliers)', xy=(mu2, pdf2.max()*1.1), xytext=(mu2+2, pdf2.max()*2.0),
    #             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
    #             fontsize=12, ha='center')

    # Customize Legend
    leg = ax.legend(loc='upper right', frameon=False, fontsize=11)
    
    # Final layout adjustments
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, transparent=True, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_gmm_illustration()