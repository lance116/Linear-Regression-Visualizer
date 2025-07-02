import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


house_data = [
    [1000, 200000],
    [1200, 240000], 
    [1500, 300000],
    [1800, 360000],
    [2000, 400000],
    [2500, 500000]
]


for size, price in house_data:
    print(f"Size: {size} sqft, Price: ${price}")

# Convert to pandas DataFrame
df = pd.DataFrame(house_data, columns=['Size', 'Price'])

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Size'], df['Price'], color='blue', s=100)
plt.xlabel('House Size (Square Feet)')
plt.ylabel('Price ($)')
plt.title('House Size vs Price')
plt.grid(True)
plt.show()

# Prepare the data for machine learning
X = df[['Size']]  # Input: house sizes (the [ ] makes it the right shape)
y = df['Price']   # Output: prices we want to predict

# Create and train the AI model
model = LinearRegression()
model.fit(X, y)  # This is where the AI learns!

print("AI training complete")
print(f"The AI learned: For every extra sq ft, price goes up by ${model.coef_[0]:.0f}")

# Make predictions for new house sizes
test_sizes = [1700, 2300, 1100]

print("\n🔮 AI Predictions:")
for size in test_sizes:
    predicted_price = model.predict([[size]])[0]
    print(f"A {size} sq ft house should cost: ${predicted_price:,.0f}")