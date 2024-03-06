import matplotlib.pyplot as plt
import numpy as np

def make_reliability_diagram(emos_dict, X, y, variances, t, n_subset = 10):
    subset_values = np.linspace(0, 1, n_subset)
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Create a figure with two subplots

    for name, model in emos_dict.items():
        distributions = model.forecast_distribution.get_distribution(X, variances)
        cdf = distributions.cdf
        cdf_values = cdf(t)

        #find the indices of cdf_values (which is a tf.tensor) that lie in between the subset_values
        indices = []
        for value in subset_values:
            indices.append(np.where(cdf_values < value)[0][-1])
        indices = np.array(indices)

        #compute the empirical probabilities based on y and the indices
        empirical_probabilities = np.zeros(n_subset)
        for i in range(n_subset - 1):
            empirical_probabilities[i] = np.mean(y[indices[i]:indices[i + 1]] < t)
        
        #compute the average cdf values for each subset
        average_cdf_values = np.zeros(n_subset)
        for i in range(n_subset - 1):
            average_cdf_values[i] = np.mean(cdf_values[indices[i]:indices[i + 1]])

        #plot the reliability diagram
        axs[0].plot(average_cdf_values, empirical_probabilities, label = name)
        #plot the refinement diagram
        axs[1].plot(average_cdf_values, np.diff(indices), label = name)  # Assuming refinement is the difference between indices

    axs[0].plot([0, 1], [0, 1], color="black", linestyle="dashed")
    axs[0].set_xlabel("Forecast probability")
    axs[0].set_ylabel("Empirical probability")
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[0].legend()

    axs[1].set_xlabel("Forecast probability")
    axs[1].set_ylabel("Refinement")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].legend()

    plt.tight_layout()
    plt.show()