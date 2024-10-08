from lib.data_loading import *
from NNs.autoencoder import *
import torch
import matplotlib.pyplot as plt 
from config.vae import VAE_config
from sklearn.manifold import TSNE
from lib.helper import *    
from matplotlib import ticker
import matplotlib as    mpl
import argparse
from darts.models import TFTModel
from config.tft import Tft_config
from sklearn.decomposition import PCA
from matplotlib import cm
import matplotlib.patches as mpatches
mpl.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['font.serif'] = ['Palatino']


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
parser.add_argument('--train', type=str, default="False", help='train or not')
args = parser.parse_args()

if args.train == "True":
    train = True
else:
    train = False
    
VAE= torch.load("models/model_vae_"+ args.data_type +"_final.pt", map_location=torch.device("cpu"))
if args.data_type == 'avg':
    y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_train.npy"), dtype=torch.float32)
    y_pred_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_train_std.npy"), dtype= torch.float32)
    y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_test.npy"), dtype=torch.float32) 
    y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_test_std.npy"), dtype= torch.float32)
    data_train = Tft_config.data_train_avg
    data_test = Tft_config.data_test_avg
    train_prarams = Tft_config.train_params_avg
    test_params = Tft_config.test_params_avg
elif args.data_type == 'std':  
    y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_train.npy"), dtype=torch.float32)
    y_pred_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_train_std.npy"), dtype= torch.float32)
    y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_test.npy"), dtype=torch.float32) 
    y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_test_std.npy"), dtype= torch.float32)
    data_train = Tft_config.data_train_std
    data_test = Tft_config.data_test_std    
    train_params = Tft_config.train_params_std
    test_params = Tft_config.test_params_std
    

#_,_, y_pred_encoded, y_pred_test_encoded = get_encoded_decoded(VAE, y_pred, y_pred_test) 

def add_2d_scatter(ax, points, points_color,config_list, trj=None, title=None):
    
    #points = points.T
    k=0
    for i  in range(0,105,1):

        label =f"{config_list[i,0]:.0f}, {config_list[i,1]:.1e}, {config_list[i,2]:.2}"
        if i < 84:
            scatter = ax.scatter(points[i,:,0], points[i,:,1], color="blue", s=25, alpha=0.5)
        else:
            scatter = ax.scatter(points[i,:,0], points[i,:,1], color="red", s=25, alpha=0.5)
        k+=1
    scatter = ax.scatter(trj[:,0], trj[:,1], color="green", s=25, alpha=0.5)
    #scatter=ax.scatter(x, y, c=points_color, s=25, alpha=0.5)
    k=0
    # for i, (x, y) in enumerate(zip(x, y)):
    #     if k <15:
    #         plt.annotate(f'{k}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='blue')
    #     elif k >= 15 and k < 100:
    #         if i%10 == 0:   
    #             plt.annotate(f'{k}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='blue')
    #     else:
    #         if k%50 == 0:   
    #             plt.annotate(f'{k}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='blue')
    #     k+=1
    #     if k==500:
    #         k=0
    #legend1 = ax.legend(*scatter.legend_elements(),
                   # loc="lower left", title="Classes")
    #ax.add_artist(legend1)
    legend_handles = [
    mpatches.Patch(color="blue", label=r'Train set'),
    mpatches.Patch(color="red", label=r'Test set'),
    mpatches.Patch(color="green", label=r'Trj')
]
    ax.legend(handles=legend_handles ,loc="lower right")
    ax.set_title(title)
    #ax.xaxis.set_major_formatter(ticker.NullFormatter())
    #ax.yaxis.set_major_formatter(ticker.NullFormatter())
def plot_2d(points, points_color, title, config_list, trj=None):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color, config_list, trj=trj)
    plt.savefig("figs/t_sne_plots/"+title+".png")

    

trj = torch.tensor(np.load("../../data_kmc/1400_1e+20_2.0/avg_trj.npy"), dtype=torch.float32)
trj = torch.reshape(trj,(1000,1,50,100))
trj.unsqueeze(0)
with torch.no_grad():
  trj_vae = VAE(trj)[2] ##encoded mu
data_test = data_test/ torch.mean(data_test)
data_train = data_train/ torch.mean(data_train)

data_train = data_train[:,:300]

data_test = data_test[:,:300]
train_data_class = np.linspace(1, 84, 84)

train_data_class = np.repeat(train_data_class, data_train.shape[1])
with torch.no_grad():
    data_train_vae = reshape_vae(data_train)
    data_train_vae = VAE(data_train_vae)[2] ## Encoding the data into the latent space
    data_train_encoded = unshape_vae(data_train_vae, data_train.shape[0], data_train.shape[1], True)

    data_test_vae = reshape_vae(data_test)
    data_test_vae = VAE(data_test_vae)[2] ## Encoding the data into the latent space

    data_test_encoded = unshape_vae(data_test_vae, data_test.shape[0], data_test.shape[1], True)

data_comp = torch.cat((data_train_vae, data_test_vae), 0).numpy()


n_components = 2    
tsne = TSNE(n_components=n_components, verbose=1,perplexity= 30,  random_state=0)
pca = PCA(n_components=n_components) 
if train:

    data_comp_tsne = tsne.fit_transform(data_comp)
    data_comp_pca = pca.fit_transform(data_comp)
    
    np.save("eval_01/data/data_comp_pca.npy", data_comp_pca)
    np.save("eval_01/data/data_comp_tsne.npy", data_comp_tsne)
    data_comp_pca = data_comp_pca.reshape(105, 300, 2)
    data_comp_tsne = data_comp_tsne.reshape(105, 300, 2)
else:
    #data_train_tsne = np.load("eval_01/data/data_train_tsne.npy")
    data_train_pca = np.load("eval_01/data/data_train_pca.npy")
    data_comp_pca = np.load("eval_01/data/data_comp_pca.npy")   
    data_comp_tsne = np.load("eval_01/data/data_comp_tsne.npy")
    data_comp_pca = data_comp_pca.reshape(105, 300, 2)
    data_comp_tsne = data_comp_tsne.reshape(105, 300, 2)

#data_trj_1_2 = data_train_tsne[:1000]
#data_train_pca = data_train_pca[7500:8000]
num_colors = 84
colors = mpl.cm.rainbow(np.linspace(0,1, 16))
config_list_train = get_config("../../data_kmc/2d_sets/train_set_lin_80_20_"+args.data_type+"_list.txt")
config_list_test = get_config("../../data_kmc/2d_sets/test_set_lin_80_20_"+args.data_type+"_list.txt")
config_list = np.vstack((config_list_train, config_list_test))
trj_pca = pca.transform(trj_vae[:])
trj_tsne = tsne.fit_transform(trj_vae[:])
#data_train_pca = data_train_pca.reshape(84, 500, 2)

plot_2d(data_comp_pca, colors, "train_test_1e20_pca", config_list, trj_pca)
plot_2d(data_comp_tsne, colors, "train_test_1e20_tsne", config_list, trj_pca)

