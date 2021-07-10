from autoencoder import fetch_data, make_model, train, latent_predict, load_and_predict, IMAGE_SIZE, NUM_CLASSES

# Fetch 'chinese_mnist' dataset and prepare data
data = fetch_data("data/chinese_mnist")

# Make and compile Encode & Decoder
model = make_model(NUM_CLASSES, IMAGE_SIZE)

# Train or load a pretrained model if it exists
epochs = 25
model = train(model, data, model_file=f'models/ae_{epochs}epochs', num_epochs=epochs)


import matplotlib.pyplot as plt

preds = load_and_predict(model, data)

idx = 11
plt.imshow(preds[idx])

img = list(data[1])[idx][1].numpy()[0]
plt.imshow(img)