import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

num_articles = 1000

article_lengths_system1 = np.random.poisson(lam=200, size=num_articles) + np.random.randint(-156, 229, size=num_articles)

article_lengths_system2 = np.random.poisson(lam=247, size=num_articles) + np.random.randint(-110, 176, size=num_articles)

kde_system1 = gaussian_kde(article_lengths_system1)
kde_system2 = gaussian_kde(article_lengths_system2)
x = np.linspace(0, max(max(article_lengths_system1), max(article_lengths_system2)), 1000)
y1 = kde_system1(x)
y2 = kde_system2(x)
plt.figure(figsize=(10, 6))

plt.plot(x, y1, alpha=0.5, color='b', label='Normal')
plt.plot(x, y2, alpha=0.5, color='r', label='With knowledge retrieval')

plt.title('Abstract Length Distribution')
plt.xlabel('Abstract Length')
plt.ylabel('Frequency')
plt.legend()

plt.grid(True)
plt.show()
