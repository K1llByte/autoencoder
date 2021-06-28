from autoencoder_test import fetch_data, make_model, train, IMAGE_SIZE, NUM_CLASSES

# Fetch 'chinese_mnist' dataset and prepare data
data = fetch_data("data/chinese_mnist")

# Make and compile Encode & Decoder
model = make_model(NUM_CLASSES, IMAGE_SIZE)

# Train or load a pretrained model if it exists
epochs=5
model = train(model, data, model_file=f'models/ae_{epochs}epochs', num_epochs=epochs)