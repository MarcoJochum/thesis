from lib.data_loading import *
from NNs.autoencoder import *
import torch
import matplotlib.pyplot as plt 
from config.vae import VAE_config
from sklearn.manifold import TSNE
from lib.helper import *    
from matplotlib import ticker

train = False

def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=25, alpha=0.5)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.savefig("figs/t_sne_plots/t_sne_train_1_trj.png")
vae = torch.load("models/model_vae_lin_lat_10.pth", map_location=torch.device("cpu"))
data_train = VAE_config.data_train
data_test = VAE_config.data_test
data_test = data_test/ torch.mean(data_test)
data_train = data_train/ torch.mean(data_train)

data_train = data_train[:,:VAE_config.n_steps]
#data_train = data_train[:,::30]

data_test = data_test[:,:VAE_config.n_steps]
train_data_class = np.linspace(0, 77, 78)

train_data_class = np.repeat(train_data_class, data_train.shape[1])
with torch.no_grad():
    data_train_vae = reshape_vae(data_train)
    data_train_vae = vae(data_train_vae)[2] ## Encoding the data into the latent space
    data_train_encoded = unshape_vae(data_train_vae, data_train.shape[0], data_train.shape[1], True)

    data_test_vae = reshape_vae(data_test)
    data_test_vae = vae(data_test_vae)[2] ## Encoding the data into the latent space

    data_test_encoded = unshape_vae(data_test_vae, data_test.shape[0], data_test.shape[1], True)

data_encoded = torch.cat((data_train_vae, data_test_vae), 0)
data_encoded = data_encoded.numpy()


n_components = 2    
tsne = TSNE(n_components=n_components, verbose=1,perplexity= 30,  random_state=0)

if train:

    data_tsne = tsne.fit_transform(data_train_vae.numpy())
    np.save("eval_01/data/data_train_tsne.npy", data_tsne)
else:
    data_tsne = np.load("eval_01/data/data_train_tsne.npy")


data_trj_1_2 = data_tsne[:1000]
breakpoint()
plot_2d(data_trj_1_2, train_data_class[:1000], "Original data")
