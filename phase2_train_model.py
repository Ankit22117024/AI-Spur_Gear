import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # For saving the scaler

# --- Task 1: Preparing the Data ---

print("--- Task 1: Preparing the Data ---")
# Load the dataset we created in Phase 1
df = pd.read_csv('expert_gear_solutions.csv')

# Separate the inputs (features) from the outputs (labels)
features = df[['Input_Power_kW', 'Input_Speed_RPM', 'Velocity_Ratio']]
labels = df[['Result_Module', 'Result_Teeth', 'Result_FaceWidth']]

# Split the data into training and testing sets (80% train, 20% test)
# This lets us test the model on data it has never seen before.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale the data. This is crucial for neural networks.
# It normalizes the data to have a mean of 0 and a standard deviation of 1,
# which helps the model train faster and more effectively.
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

print("Data successfully loaded, split, and scaled.")
print(f"Training set size: {X_train_scaled.shape[0]} samples")
print(f"Testing set size: {X_test_scaled.shape[0]} samples")

# --- Task 2: Building and Training the Model ---

print("\n--- Task 2: Building and Training the Model ---")
# Define the neural network architecture
model = tf.keras.Sequential([
    # Input layer: needs to know the shape of our input features
    tf.keras.layers.Dense(64, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    
    # Hidden layers: these layers find complex patterns in the data
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    
    # Output layer: 3 neurons for our 3 outputs (module, teeth, width)
    # No activation function needed for regression problems
    tf.keras.layers.Dense(3)
])

# Compile the model with an optimizer and a loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("Training the model...")
history = model.fit(
    X_train_scaled, 
    y_train_scaled,
    epochs=100,
    validation_split=0.2, # Use 20% of training data for validation
    verbose=0 # Set to 1 to see live progress
)

print("Model training complete.")

# --- Task 3: Evaluating and Saving the Model ---

print("\n--- Task 3: Evaluating and Using the Model ---")
# Evaluate the model's performance on the unseen test data
loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f"Model performance on Test Data (Loss - MSE): {loss:.4f}")

# Let's make a sample prediction
sample_input = [[10, 1200, 3]] # 10 kW, 1200 RPM, Ratio 3
print(f"\nMaking a prediction for sample input: {sample_input[0]}")

# We must scale our sample input using the SAME scaler from training
sample_input_scaled = scaler_X.transform(sample_input)

# Get the prediction (which will be in scaled format)
prediction_scaled = model.predict(sample_input_scaled)

# We must inverse_transform the prediction to get human-readable values
prediction = scaler_y.inverse_transform(prediction_scaled)

print("--- Prediction Result ---")
print(f"Predicted Module: {prediction[0][0]:.2f} mm")
print(f"Predicted Teeth: {prediction[0][1]:.0f}")
print(f"Predicted Face Width: {prediction[0][2]:.2f} mm")

# Save the trained model and the scalers for Phase 3
model.save('gear_predictor_model.h5')
joblib.dump(scaler_X, 'scaler_X.joblib')
joblib.dump(scaler_y, 'scaler_y.joblib')

print("\n----------------------------------------------------")
print("✅ Phase 2 Complete!")
print("Model and scalers have been saved.")
print("We are now ready to build the user interface in Phase 3.")