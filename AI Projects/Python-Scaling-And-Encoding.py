import sklearn
print(sklearn.__version__)


import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Sample Data
data = pd.DataFrame({'Age':[25,30,35],'Gender':['M','F','M']})

# Scaling numerical column

scaler = StandardScaler()
data['Age_scaled'] = scaler.fit_transform(data[['Age']])

# Encoding categorical column
encoder = OneHotEncoder(sparse=False)
gender_encoded = encoder.fit_transform(data[['Gendder']])