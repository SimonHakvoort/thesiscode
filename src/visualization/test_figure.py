import matplotlib.pyplot as plt

def test(x):
    return 2 * x + 3

x = range(-10, 11)

y = [test(i) for i in x]

plt.plot(x, y)

plt.show()