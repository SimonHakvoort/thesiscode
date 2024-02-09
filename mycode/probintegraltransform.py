import matplotlib.pyplot as plt

def MakeProbabilityIntegralTransFormHistrogram(cdf, realizations, number_of_bins = 10):
    plt_values = cdf(realizations)

    plt.hist(plt_values, bins=number_of_bins, density=True, alpha=0.1, color='g', edgecolor='black', range=(0,1), align='mid')
    plt.plot([0, 1], [1, 1], "r--")
    #plt.xlim(0, 1)
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.show()