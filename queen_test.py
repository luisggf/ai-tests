import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Example data (replace this with your actual data)
data = {
    'Generations': [58141, 681, 166, 642, 3242,
                    45, 5, 56, 4, 37,
                    9, 41, 1, 2, 4,
                    101, 24, 588, 89, 1464,
                    751, 3111, 232, 6180, 701,
                    54, 10, 4, 4, 9,
                    753, 6033, 2088, 121, 27,
                    392, 150, 165, 755, 53,
                    730, 563, 15, 92, 122],

    'Time': [455.68, 4.49, 1.05, 4.87, 24.06,
             2.02, 0.21, 2.06, 0.17, 1.35,
             0.79, 3.49, 0.13, 0.26, 0.33,
             0.92, 0.15, 5.08, 0.64, 10.78,
             56.06, 144.93, 9.1, 288.15, 48.12,
             5.11, 1.12, 0.42, 0.41, 0.97,
             5.93, 51.02, 15.43, 0.91, 0.20,
             18.05, 7.71, 7.68, 38.07, 2.35,
             88.71, 68.07, 1.78, 10.82, 13.88],

    # Repeat each value 5 times
    'Queens': [8, 8, 8, 8, 8,
               8, 8, 8, 8, 8,
               8, 8, 8, 8, 8,
               10, 10, 10, 10, 10,
               10, 10, 10, 10, 10,
               10, 10, 10, 10, 10,
               12, 12, 12, 12, 12,
               12, 12, 12, 12, 12,
               12, 12, 12, 12, 12],
    'MutationRate': [0.05] * 45,  # Repeat the value 15 times
}
df = pd.DataFrame(data)

# Scatter plot
plt.figure(figsize=(10, 6))

for queens in df['Queens'].unique():
    subset = df[df['Queens'] == queens]
    plt.scatter(subset['Generations'], subset['Time'],
                label=f'{queens} Queens')

plt.title('Scatter Plot of Time vs. Generations for Mutation Rate = 0.05')
plt.xlabel('Generations')
plt.ylabel('Time (seconds)')
plt.legend(title='Number of Queens')


average_time = df.groupby(['Queens', 'Generations']).mean()['Time'].unstack().T
average_time.plot(kind='bar', width=0.8, figsize=(8, 6), colormap='viridis')
plt.title('Average Time for Each Number of Queens and Generations')
plt.xlabel('Generations')
plt.ylabel('Average Time (seconds)')

plt.tight_layout()
plt.show()
