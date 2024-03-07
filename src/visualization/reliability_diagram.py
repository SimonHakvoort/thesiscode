import matplotlib.pyplot as plt
import numpy as np

def make_reliability_and_refinement_diagram(emos_dict, X, y, variances, t, n_subset = 11):
    subset_values = np.linspace(0, 1, n_subset)
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Create a figure with two subplots

    for name, model in emos_dict.items():
        distributions = model.forecast_distribution.get_distribution(X, variances)
        cdf = distributions.cdf
        cdf_values = cdf(t).numpy()

        empirical_probabilities = np.zeros(n_subset - 1)
        average_cdf_values = np.zeros(n_subset - 1)
        counts = np.zeros(n_subset - 1)
        for i in range(0, len(cdf_values)):
            for j in range(0, n_subset - 1):
                if cdf_values[i] >= subset_values[j] and cdf_values[i] < subset_values[j + 1]:
                    counts[j] += 1
                    average_cdf_values[j] += cdf_values[i]
                    if y[i] < t:
                        empirical_probabilities[j] += 1

        axs[0].plot(average_cdf_values / counts, empirical_probabilities / counts, 'o-', label = name)
        
        # #in ax[1] we will plot the refinement diagram, which contains the counts for each bin
        axs[1].plot(subset_values[0:-1], counts / len(cdf_values), label = name)

        



    axs[0].plot([0, 1], [0, 1], color="black", linestyle="dashed")
    axs[0].set_xlabel("Forecast probability")
    axs[0].set_ylabel("Empirical probability")
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[0].legend()

    axs[1].set_xlabel("Forecast probability")
    axs[1].set_ylabel("Count")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def make_reliability_diagram(emos_dict, X, y, variances, t, n_subset = 11):
    subset_values = np.linspace(0, 1, n_subset)

    for name, model in emos_dict.items():
        distributions = model.forecast_distribution.get_distribution(X, variances)
        cdf = distributions.cdf
        cdf_values = cdf(t).numpy()

        empirical_probabilities = np.zeros(n_subset - 1)
        average_cdf_values = np.zeros(n_subset - 1)
        counts = np.zeros(n_subset - 1)
        for i in range(0, len(cdf_values)):
            for j in range(0, n_subset - 1):
                if cdf_values[i] >= subset_values[j] and cdf_values[i] < subset_values[j + 1]:
                    counts[j] += 1
                    average_cdf_values[j] += cdf_values[i]
                    if y[i] < t:
                        empirical_probabilities[j] += 1

        plt.plot(average_cdf_values / counts, empirical_probabilities / counts, 'o-', label = name)

    plt.plot([0, 1], [0, 1], color="black", linestyle="dashed")
    plt.xlabel("Forecast probability")
    plt.ylabel("Empirical probability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()