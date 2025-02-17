import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Function to calculate squared error
def calculate_error(target, output):
    return (target - output) ** 2

# Function to compute gradient of the error w.r.t weight
def gradient_error(target, output, input_value):
    return -2 * (target - output) * output * (1 - output) * input_value

# Function to update weight
def update_weight(weight, learning_rate, gradient):
    return weight - learning_rate * gradient

# Function to update bias
def update_bias(bias, learning_rate, bias_gradient):
    return bias - learning_rate * bias_gradient

train_data = [
    {"x1": 96440, "target": 4.7, "age": 15, "heart_rate": 77},
    {"x1": 71759, "target": 4.7, "age": 73, "heart_rate": 60},
    {"x1": 91330, "target": 2.4, "age": 60, "heart_rate": 77},
    {"x1": 90947, "target": 3.2, "age": 44, "heart_rate": 69},
    {"x1": 62154, "target": 3.4, "age": 46, "heart_rate": 60},
    {"x1": 93796, "target": 3.8, "age": 41, "heart_rate": 75},
    {"x1": 92504, "target": 4.7, "age": 71, "heart_rate": 79},
    {"x1": 63803, "target": 2.4, "age": 17, "heart_rate": 60},
    {"x1": 91533, "target": 3.8, "age": 68, "heart_rate": 57},
    {"x1": 67184, "target": 3.3, "age": 53, "heart_rate": 64},
    {"x1": 58124, "target": 3.0, "age": 33, "heart_rate": 71},
    {"x1": 71956, "target": 2.4, "age": 31, "heart_rate": 80},
    {"x1": 99513, "target": 4.2, "age": 32, "heart_rate": 97},
    {"x1": 63031, "target": 3.4, "age": 48, "heart_rate": 90},
    {"x1": 94378, "target": 5.2, "age": 33, "heart_rate": 63},
    {"x1": 80129, "target": 4.1, "age": 52, "heart_rate": 86},
    {"x1": 98111, "target": 3.7, "age": 24, "heart_rate": 92},
    {"x1": 68401, "target": 8.9, "age": 55, "heart_rate": 77},
    {"x1": 87541, "target": 2.4, "age": 58, "heart_rate": 93},
    {"x1": 104118, "target": 2.4, "age": 38, "heart_rate": 69},
    {"x1": 82888, "target": 3.9, "age": 52, "heart_rate": 90},
    {"x1": 90888, "target": 4.4, "age": 47, "heart_rate": 98},
    {"x1": 91192, "target": 4.0, "age": 21, "heart_rate": 60},
    {"x1": 61281, "target": 4.3, "age": 61, "heart_rate": 60},
    {"x1": 98640, "target": 2.4, "age": 53, "heart_rate": 93},
    {"x1": 102176, "target": 2.4, "age": 57, "heart_rate": 64},
    {"x1": 74204, "target": 6.9, "age": 46, "heart_rate": 103},
    {"x1": 94380, "target": 9.4, "age": 37, "heart_rate": 71},
    {"x1": 106545, "target": 2.4, "age": 73, "heart_rate": 71},
    {"x1": 91330, "target": 4.6, "age": 15, "heart_rate": 77},
    {"x1": 91330, "target": 5.0, "age": 84, "heart_rate": 77},
    {"x1": 107632, "target": 4.8, "age": 64, "heart_rate": 84},
    {"x1": 51735, "target": 3.4, "age": 72, "heart_rate": 84},
    {"x1": 56543, "target": 5.2, "age": 59, "heart_rate": 66},
    {"x1": 86491, "target": 5.8, "age": 49, "heart_rate": 60}
]

test_data = [
    {"x1": 95199, "target": 4.2, "age": 46, "heart_rate": 84},
    {"x1": 96337, "target": 4.0, "age": 59, "heart_rate": 74},
    {"x1": 95251, "target": 5.4, "age": 64, "heart_rate": 93},
    {"x1": 51776, "target": 4.5, "age": 47, "heart_rate": 35},
    {"x1": 105055, "target": 4.4, "age": 25, "heart_rate": 65},
    {"x1": 76561, "target": 6.1, "age": 44, "heart_rate": 74},
    {"x1": 52788, "target": 7.1, "age": 40, "heart_rate": 67}
]

# Prepare data for normalization
x1_values = [sample["x1"] for sample in train_data + test_data]
age_values = [sample["age"] for sample in train_data + test_data]
heart_rate_values = [sample["heart_rate"] for sample in train_data + test_data]
target_values = [sample["target"] for sample in train_data + test_data]

# Fit MinMaxScaler
scaler_x1 = MinMaxScaler()
scaler_age = MinMaxScaler()
scaler_heart_rate = MinMaxScaler()
scaler_target = MinMaxScaler()

# Reshape and normalize features and targets
x1_values = np.array(x1_values).reshape(-1, 1)
age_values = np.array(age_values).reshape(-1, 1)
heart_rate_values = np.array(heart_rate_values).reshape(-1, 1)
target_values = np.array(target_values).reshape(-1, 1)

scaler_x1.fit(x1_values)
scaler_age.fit(age_values)
scaler_heart_rate.fit(heart_rate_values)
scaler_target.fit(target_values)

# Apply normalization to training and testing data
for sample in train_data + test_data:
    sample["x1"] = scaler_x1.transform(np.array(sample["x1"]).reshape(-1, 1))[0][0]
    sample["age"] = scaler_age.transform(np.array(sample["age"]).reshape(-1, 1))[0][0]
    sample["heart_rate"] = scaler_heart_rate.transform(np.array(sample["heart_rate"]).reshape(-1, 1))[0][0]
    sample["target"] = scaler_target.transform(np.array(sample["target"]).reshape(-1, 1))[0][0]

# Initialize weights and bias
weight_x1 = 0.1
weight_age = 0.1
weight_heart_rate = 0.1
bias = 0.1
learning_rate = 0.00001
epochs = 1000000  # Increased for demonstration

# Early stopping parameters
patience = 10
best_loss = float('inf')
patience_count = 0

# Lists to store loss per epoch
train_losses = []
test_losses = []

# Training loop
for epoch in range(epochs):
    train_loss = 0
    test_loss = 0

    # Training data processing
    for sample in train_data:
        input_value_x1 = sample["x1"]
        input_value_age = sample["age"]
        input_value_heart_rate = sample["heart_rate"]
        target = sample["target"]
        z = input_value_x1 * weight_x1 + input_value_age * weight_age + input_value_heart_rate * weight_heart_rate + bias
        prediction = sigmoid(z)
        error = calculate_error(target, prediction)
        train_loss += error

        # Update weights and bias
        gradient_x1 = gradient_error(target, prediction, input_value_x1)
        gradient_age = gradient_error(target, prediction, input_value_age)
        gradient_heart_rate = gradient_error(target, prediction, input_value_heart_rate)
        weight_x1 = update_weight(weight_x1, learning_rate, gradient_x1)
        weight_age = update_weight(weight_age, learning_rate, gradient_age)
        weight_heart_rate = update_weight(weight_heart_rate, learning_rate, gradient_heart_rate)
        bias_gradient = -2 * (target - prediction) * prediction * (1 - prediction)
        bias = update_bias(bias, learning_rate, bias_gradient)

    # Calculate average training loss
    train_loss /= len(train_data)

    # Testing data processing
    for sample in test_data:
        input_value_x1 = sample["x1"]
        input_value_age = sample["age"]
        input_value_heart_rate = sample["heart_rate"]
        target = sample["target"]
        z = input_value_x1 * weight_x1 + input_value_age * weight_age + input_value_heart_rate * weight_heart_rate + bias
        prediction = sigmoid(z)
        error = calculate_error(target, prediction)
        test_loss += error

    # Calculate average testing loss
    test_loss /= len(test_data)

    # Store losses for plotting
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # Check for improvement
    if test_loss < best_loss:
        best_loss = test_loss
        patience_count = 0  # Reset patience counter
    else:
        patience_count += 1  # Increment patience counter

    # Early stopping condition
    if patience_count > patience:
        print(f'Stopping early at epoch {epoch + 1}')
        break

# Plotting the train and test losses per epoch
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses, label='Valid Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Valid Loss per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('GULA_DARAH_loss.jpg', format='jpg')
plt.show()

# Create a DataFrame to save to Excel
epoch_data = {
    'Epoch': np.arange(1, len(train_losses) + 1),
    'Train Loss': train_losses,
    'Test Loss': test_losses
}

df = pd.DataFrame(epoch_data)

# Save to Excel
df.to_excel('GULA_DARAH.xlsx', index=False)

# Final values
print(f'Optimal Weight for x1: {weight_x1}')
print(f'Optimal Weight for age: {weight_age}')
print(f'Optimal Weight for heart rate: {weight_heart_rate}')
print(f'Optimal Bias: {bias}')
print(f'Final Train Loss: {train_loss}')
print(f'Final Test Loss: {test_loss}')
