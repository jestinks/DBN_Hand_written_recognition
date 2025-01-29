import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import cv2
import gradio as gr
import matplotlib.pyplot as plt

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, (n_visible, n_hidden))
        self.hidden_bias = np.zeros(n_hidden)
        self.visible_bias = np.zeros(n_visible)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, visible_probabilities):
        hidden_activations = np.dot(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self.sigmoid(hidden_activations)
        return hidden_probabilities, (hidden_probabilities > np.random.random(hidden_probabilities.shape))

    def sample_visible(self, hidden_probabilities):
        visible_activations = np.dot(hidden_probabilities, self.weights.T) + self.visible_bias
        visible_probabilities = self.sigmoid(visible_activations)
        return visible_probabilities, (visible_probabilities > np.random.random(visible_probabilities.shape))

    def train(self, data, epochs=10, batch_size=100):
        for epoch in range(epochs):
            error = 0
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                pos_hidden_probs, pos_hidden_states = self.sample_hidden(batch)
                pos_associations = np.dot(batch.T, pos_hidden_probs)
                neg_visible_probs, _ = self.sample_visible(pos_hidden_states)
                neg_hidden_probs, _ = self.sample_hidden(neg_visible_probs)
                neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
                self.weights += self.learning_rate * ((pos_associations - neg_associations) / batch_size)
                self.hidden_bias += self.learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)
                self.visible_bias += self.learning_rate * np.mean(batch - neg_visible_probs, axis=0)
                error += np.mean((batch - neg_visible_probs) ** 2)
            print(f'Epoch {epoch+1}, Reconstruction error: {error/len(data):.4f}')

class DBN:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.rbm_layers = [RBM(layers[i], layers[i+1], learning_rate) for i in range(len(layers)-1)]

    def get_hidden_and_reconstruction(self, data):
        hidden = data
        for rbm in self.rbm_layers:
            hidden, _ = rbm.sample_hidden(hidden)
        reconstruction = hidden
        for rbm in reversed(self.rbm_layers):
            reconstruction, _ = rbm.sample_visible(reconstruction)
        return hidden, reconstruction

    def train(self, data, epochs=10, batch_size=100):
        input_data = data
        for i, rbm in enumerate(self.rbm_layers):
            print(f'\nTraining RBM layer {i+1}')
            rbm.train(input_data, epochs, batch_size)
            input_data, _ = rbm.sample_hidden(input_data)

def load_mnist_data():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    return x_train, y_train

def save_dbn_model(dbn, filename='dbn_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(dbn, f)

def load_dbn_model(filename='dbn_model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def preprocess_custom_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 784)
    return img

def process_custom_input(image, model):
    processed_img = preprocess_custom_image(image)
    hidden, reconstruction = model.get_hidden_and_reconstruction(processed_img)
    return processed_img.reshape(28, 28), hidden.reshape(10, 10), reconstruction.reshape(28, 28)

def train_model():
    x_train, y_train = load_mnist_data()
    dbn = DBN([784, 500, 250, 100], learning_rate=0.01)
    dbn.train(x_train, epochs=6, batch_size=128)
    save_dbn_model(dbn)
    return "Model trained and saved successfully!"

def predict(image):
    if os.path.exists('dbn_model.pkl'):
        dbn = load_dbn_model()
    else:
        return "Model not found. Train the model first!"
    original, hidden, reconstructed = process_custom_input(image, dbn)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original, cmap='gray'); axes[0].set_title('Original')
    axes[1].imshow(hidden, cmap='gray'); axes[1].set_title('Hidden Representation')
    axes[2].imshow(reconstructed, cmap='gray'); axes[2].set_title('Reconstructed')
    plt.close()
    return fig

def display_reconstructed_digits():
    if os.path.exists('dbn_model.pkl'):
        dbn = load_dbn_model()
    else:
        return "Model not found. Train the model first!"
    x_train, y_train = load_mnist_data()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        index = np.where(y_train == i)[0][0]
        sample = x_train[index:index+1]
        _, reconstruction = dbn.get_hidden_and_reconstruction(sample)
        ax = axes[i // 5, i % 5]
        ax.imshow(reconstruction.reshape(28, 28), cmap='gray')
        ax.set_title(f'Reconstructed {i}')
        ax.axis('off')
    plt.close()
    return fig

def interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Deep Belief Network for Handwritten Digit Recognition")
        train_button = gr.Button("Train Model")
        train_output = gr.Textbox(label="Training Status")
        train_button.click(train_model, outputs=train_output)
        gr.Markdown("## Upload an Image for Prediction")
        image_input = gr.Image(type="pil")
        predict_button = gr.Button("Predict")
        image_output = gr.Plot()
        predict_button.click(predict, inputs=image_input, outputs=image_output)
        gr.Markdown("## Display Reconstructed Digits (0-9)")
        reconstruct_button = gr.Button("Show Reconstructed Digits")
        reconstruct_output = gr.Plot()
        reconstruct_button.click(display_reconstructed_digits, outputs=reconstruct_output)
    demo.launch()

if __name__ == "__main__":
    interface()
