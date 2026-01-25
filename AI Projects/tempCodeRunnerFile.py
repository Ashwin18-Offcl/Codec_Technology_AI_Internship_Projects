from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Model
model = Sequential([
    Input(shape=(2,)),
    Dense(8, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X, y, epochs=1000, batch_size=1, verbose=0)

# Predict
predictions = model.predict(X)
binary_output = (predictions > 0.5).astype(int)

print("Predictions:\n", predictions)
print("Binary Output:\n", binary_output)
