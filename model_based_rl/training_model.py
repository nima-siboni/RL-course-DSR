"""Here we train a DNN with the dataset which is created from the interactions with the env.

We assume that the data is available as a pandas DataFrame with the following columns:
- state: The state of the environment.
- action: The action taken by the data_creator.
- reward: The reward received by the data_creator.
- next_state: The next state of the environment.
- terminated: Whether the episode is terminated.
- truncated: Whether the episode is truncated.

We use a simple DNN with a single hidden layer to predict the next state based on the state and
action. The loss function is the mean squared error between the predicted next state and the actual
next state. We implement the model in keras and train it for 100 epochs, then save it.

"""
# 0. Import necessary libraries
import keras
import numpy as np
import pandas as pd
from utils import create_and_return_a_model

# 1. Load the data set
list_of_data_files = [
    "data_sets/data_set_0_rounds.pkl",
    "data_sets/data_set_10_rounds.pkl",
    "data_sets/data_set_20_rounds.pkl",
    "data_sets/data_set_30_rounds.pkl",
    "data_sets/data_set_40_rounds.pkl",
]

data = pd.concat([pd.read_pickle(file) for file in list_of_data_files])
# print number of raw of data
print(data.shape)
# 1.1. Extract the state, action, and next_state columns
state = np.array(data["state"].tolist())
action = np.array(data["action"].tolist())

# 1.2. Extract the next_state column
next_state = np.array(data["next_state"].tolist())

# 2. Define the model
model = create_and_return_a_model([64, 64, 64])
# 3. Train the model
model.fit([state, action], next_state, epochs=40)

# 4. Save the model
model.save("nn_models/model_based_rl_model_good.h5")
print("Model saved.")
# 5. Load the model
model = keras.models.load_model("nn_models/model_based_rl_model_good.h5")

# 6. Predict the next state
state = np.array([[0.0, 0.0, 0.0, 0.0]])
action = np.array([[0]])
next_state = model.predict([state, action])
print(f"Predicted next state: {next_state}")


# 10. Print the model configuration
print(model.get_config())
