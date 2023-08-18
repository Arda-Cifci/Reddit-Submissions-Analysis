import matplotlib.pyplot as plt
import seaborn as sns


def plot_mean_bar_graph_3candidates(candidate1, candidate2, candidate3, title, xlabel, ylabel, save_path):

    sns.set()
    
    means = [candidate1.mean(), candidate2.mean(), candidate3.mean()]
    fig, ax = plt.subplots()

    bars = ax.bar([0, 1, 2], means, color=['skyblue', 'lightcoral'], capsize=10)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(xlabel)

    # Add labels on the bars
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                f'{means[i]:.2f}', ha='center', va='bottom')

    plt.savefig(save_path)
