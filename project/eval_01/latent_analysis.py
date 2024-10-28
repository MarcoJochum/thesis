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
import umap
mpl.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['font.serif'] = ['Palatino']


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
args = parser.parse_args()


    
VAE= torch.load("models/model_vae_"+ args.data_type +"_final.pt", map_location=torch.device("cpu"))
if args.data_type == 'avg':
    y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_train.npy"), dtype=torch.float32)
    y_pred_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_train_std.npy"), dtype= torch.float32)
    y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_test.npy"), dtype=torch.float32) 
    y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_test_std.npy"), dtype= torch.float32)
    y_pred_trj_2 = np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_1400_1e+20_2.0.npy")
    y_pred_trj_15 = np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_1400_1e+20_1.5.npy")
    y_pred_trj_1 = np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_500_steps_"+ args.data_type+ "/y_pred_1400_1e+20_1.0.npy")
    data_train = Tft_config.data_train_avg
    data_test = Tft_config.data_test_avg
    train_prarams = Tft_config.train_params_avg
    test_params = Tft_config.test_params_avg
elif args.data_type == 'std':  
    y_pred = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_train.npy"), dtype=torch.float32)
    y_pred_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_train_std.npy"), dtype= torch.float32)
    y_pred_test = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_test.npy"), dtype=torch.float32) 
    y_pred_test_std = torch.tensor(np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_test_std.npy"), dtype= torch.float32)
    y_pred_trj_2 = np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_1400_1e+20_2.0.npy")
    y_pred_trj_15 = np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_1400_1e+20_1.5.npy")
    y_pred_trj_1 = np.load("../../data_kmc/2d_results/lin_time/study_5_scaled_sv_300_steps_"+ args.data_type+ "/y_pred_1400_1e+20_1.0.npy")
    data_train = Tft_config.data_train_std
    data_test = Tft_config.data_test_std    
    train_params = Tft_config.train_params_std
    test_params = Tft_config.test_params_std
    

_,_, y_pred_encoded, y_pred_test_encoded = get_encoded_decoded(VAE, y_pred, y_pred_test) 


trj_2 = torch.tensor(np.load("../../data_kmc/2d_high_c/1400_1e+20_2.0/"+ args.data_type+"_trj.npy"), dtype=torch.float32)
trj_15 = torch.tensor(np.load("../../data_kmc/2d_high_c/1400_1e+20_1.5/"+ args.data_type+"_trj.npy"), dtype=torch.float32)  
trj_1 = torch.tensor(np.load("../../data_kmc/2d_high_c/1400_1e+20_1.0/"+ args.data_type+"_trj.npy"), dtype=torch.float32)
trj_15 = torch.reshape(trj_15,(1000,1,50,100))
trj_15.unsqueeze(0) 
trj_1 = torch.reshape(trj_1,(1000,1,50,100))
trj_1.unsqueeze(0)
trj_2 = torch.reshape(trj_2,(1000,1,50,100))
trj_2.unsqueeze(0)


with torch.no_grad():
  trj_2_vae = VAE(trj_2)[2] ##encoded mu
  trj_15_vae = VAE(trj_15)[2]
  trj_1_vae = VAE(trj_1)[2]


data_test = data_test/ torch.mean(data_test)
data_train = data_train/ torch.mean(data_train)

data_train = data_train[:,:300]
trj_2_vae = trj_2_vae[:300]
trj_15_vae = trj_15_vae[:300]
trj_1_vae = trj_1_vae[:300]
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
tsne = TSNE(n_components=n_components, verbose=1,perplexity= 50,  random_state=0)
pca = PCA(n_components=n_components) 
umap = umap.UMAP(n_components=n_components, n_neighbors=10, min_dist=0.1, metric='euclidean')
##t-sne has to be fitted to all the data plotted
data_comp_tsne =np.vstack((data_comp, trj_1_vae,trj_15_vae,trj_2_vae))
##fit
data_comp_tsne = tsne.fit_transform(data_comp_tsne)
data_comp_pca = pca.fit_transform(data_comp)
#data_comp_umap = umap.fit_transform(data_comp)


data_comp_pca = data_comp_pca.reshape(105, 300, 2)
#data_comp_umap = data_comp_umap.reshape(105, 300, 2)
data_comp_tsne = data_comp_tsne.reshape(108, 300, 2)
##Transform
y_pred_pca = pca.transform(y_pred_encoded)
y_pred_test_pca = pca.transform(y_pred_test_encoded)
y_pred_trj_2_pca = pca.transform(y_pred_trj_2)
y_pred_trj_15_pca = pca.transform(y_pred_trj_15)
y_pred_trj_1_pca = pca.transform(y_pred_trj_1)
y_pred_comp = np.vstack((y_pred_pca, y_pred_test_pca))  
breakpoint()
y_pred_comp = y_pred_comp.reshape(105, 995, 2)


np.save("eval_01/data/pred_comp_pca_"+args.data_type+".npy", y_pred_comp)
np.save("eval_01/data/data_comp_pca_"+args.data_type+".npy", data_comp_pca)

np.save("eval_01/data/data_comp_tsne_"+args.data_type+".npy", data_comp_tsne)
#np.save("eval_01/data/data_comp_umap_"+args.data_type+".npy", data_comp_umap)

trj_2_pca = pca.transform(trj_2_vae)
trj_15_pca = pca.transform(trj_15_vae)
trj_1_pca = pca.transform(trj_1_vae)
trj_2_pca = trj_2_pca.reshape(300, 2)
trj_15_pca = trj_15_pca.reshape(300, 2)
trj_1_pca = trj_1_pca.reshape(300, 2)


np.save("eval_01/data/trj_2_pca_"+args.data_type+".npy", trj_2_pca)
np.save("eval_01/data/trj_15_pca_"+args.data_type+".npy", trj_15_pca)
np.save("eval_01/data/trj_1_pca_"+args.data_type+".npy", trj_1_pca)
    
np.save("eval_01/data/y_pred_trj_2_pca_"+args.data_type+".npy", y_pred_trj_2_pca)
np.save("eval_01/data/y_pred_trj_15_pca_"+args.data_type+".npy", y_pred_trj_15_pca)
np.save("eval_01/data/y_pred_trj_1_pca_"+args.data_type+".npy", y_pred_trj_1_pca)
    
    


