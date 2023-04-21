import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def build_recommendation_system(data):
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2)

    # Create the user embedding input
    user_input = Input(shape=[1])
    user_embedding = Embedding(input_dim=np.max(data["user_id"]) + 1, output_dim=5)(user_input)
    user_vec = Flatten()(user_embedding)

    # Create the item embedding input
    item_input = Input(shape=[1])
    item_embedding = Embedding(input_dim=np.max(data["item_id"]) + 1, output_dim=5)(item_input)
    item_vec = Flatten()(item_embedding)

    # Calculate the dot product of the user and item embeddings
    prod = Dot(axes=1)([user_vec, item_vec])

    # Combine the inputs and outputs into a model
    model = Model(inputs=[user_input, item_input], outputs=[prod])

    # Compile the model with mean squared error loss and the Adam optimizer
    model.compile(loss='mse', optimizer='adam')

    # Create a tensorboard callback for monitoring training
    tensorboard_callback = TensorBoard(log_dir='/logs', histogram_freq=1)

    # Train the model
    model.fit([train_data['user_id'], train_data['item_id']], train_data['rating'], 
              validation_data=([test_data['user_id'], test_data['item_id']], test_data['rating']),
              epochs=15, callbacks=[tensorboard_callback])

    # Evaluate the model
    predictions = model.predict([test_data['user_id'], test_data['item_id']])
    rounded_predictions = np.round(predictions)
    acc = np.round((accuracy_score(test_data['rating'], rounded_predictions)),2)
    print(f"Accuracy: {acc}")
    
    return model, acc
