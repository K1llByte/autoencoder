import math

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

##############################################################

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input)
    fig = plt.figure(figsize=(16,256))

    for i in range(predictions.shape[0]):
        plt.subplot(1, 16, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    
    
    
def plot_scatter(x,y,train_Y):
    cmap = colors.ListedColormap(['black', 'darkred', 'darkblue', 'darkgreen', 'yellow', 'brown', 'purple', 'lightgreen', 'red', 'lightblue'])
    bounds=[0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,8.5,9.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(12,10))
    ax = fig.gca()
    ax.set_aspect('equal')
    plt.scatter(x, y, c = train_Y, cmap=cmap, s = 1, norm=norm)
    plt.colorbar()
    plt.gca().add_patch(Rectangle((-2,-2), 4,4, linewidth=2, edgecolor='r', facecolor='none'))   
    plt.show()
    
    
# assumes len samples is a perfect square
# def show_samples(samples):
    
#     k = int(math.sqrt(len(samples)))
#     fig = plt.figure(figsize=(k,k))
    
#     for i in range(len(samples)):
#         plt.subplot(k, k, i+1)
#         plt.imshow(samples[i].reshape(28,28), cmap='gray')
#         plt.axis('off')

def show_samples(samples):
    
    k = int(math.sqrt(len(samples)))
    fig = plt.figure(figsize=(k,k))
    
    for i in range(len(samples)):
        plt.subplot(k, k, i+1)
        plt.imshow(np.asarray(samples)[i, :, :, 0], cmap='gray')
        plt.axis('off')