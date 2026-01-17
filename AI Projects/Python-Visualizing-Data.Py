import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = pd.DataFrame({
    'Age': [22, 25, 30, 35, 40, 45],
    'Salary': [30000, 35000, 50000, 60000, 70000, 80000]
})

# Histogram
plt.hist(data['Age'], bins=5)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Correlation Heatmap
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
