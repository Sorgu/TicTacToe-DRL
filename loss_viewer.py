import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def scatter_plot(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            data.extend([float(x) for x in row if x != ""])
    x = range(len(data))
    plt.scatter(x, data)
    plt.xlabel('Index')
    plt.ylabel('Loss')
    plt.title(file)
    chunk_size = 1000
    num_chunks = len(data) // chunk_size
    averages = [np.mean(data[i * chunk_size:(i + 1) * chunk_size]) for i in range(num_chunks)]
    midpoints = [i * chunk_size + chunk_size // 2 for i in range(num_chunks)]
    plt.plot(midpoints, averages, 'r--o', label='Average Value')
    """
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        plt.axhline(y=averages[i], xmin=start_index / len(data), xmax=end_index / len(data),
                    color='r', linestyle='--', label=f'Average (Chunk {i + 1})' if i == 0 else '')
    """
    plt.show()


scatter_plot("loss-history/loss-model1.12.csv")