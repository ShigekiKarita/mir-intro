import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
df = pd.read_csv('result.csv', index_col=0)
df.plot()
plt.savefig("plot.svg")
