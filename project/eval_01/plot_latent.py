
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from lib.helper import *
import matplotlib.patches as mpatches   
import matplotlib.ticker as ticker
import matplotlib as mpl
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
parser.add_argument('--train', type=str, default="False", help='train or not')
args = parser.parse_args()

data_comp_pca = np.load("eval_01/data/data_comp_pca_"+args.data_type+".npy")   
data_comp_tsne = np.load("eval_01/data/data_comp_tsne_"+args.data_type+".npy")
#data_comp_umap = np.load("eval_01/data/data_comp_umap_"+args.data_type+".npy")
pred_comp_pca = np.load("eval_01/data/pred_comp_pca_"+args.data_type+".npy")
config_list = np.load("../../data_kmc/2d_sets/config_list_lin_80_20_"+args.data_type+".npy")

trj_pca= np.load("eval_01/data/trj_pca_avg.npy")
pred_trj_pca = np.load("eval_01/data/y_pred_trj_pca_avg.npy")
pred_comp_pca = np.reshape(pred_comp_pca, (105, 1000, 2))
fig, ax = plt.subplots(2, 2, figsize=(10, 10))


## Scatter plot for two configs
c1 = 1
c2 = 87    
c3= 17

ax[0,0].scatter(data_comp_pca[c1,:,0], data_comp_pca[c1,:,1], color="blue", s=25, alpha=0.5,label=f"{config_list[c1,0]:.0f}, {config_list[c1,1]:.1e}, {config_list[c1,2]:.2}")
ax[0,0].scatter(data_comp_pca[c2,:,0], data_comp_pca[c2,:,1], color="red", s=25, alpha=0.5, label=f'{config_list[c2,0]:.0f}, {config_list[c2,1]:.1e}, {config_list[c2,2]:.2}')
k=1
for i, (x, y) in enumerate(zip(data_comp_pca[c2,:,0], data_comp_pca[c2,:,1])):
    if k <11:
        ax[0,0].annotate(f'{k}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black')
    
    k+=1
    if k==500:
        k=0
#Prediction data

ax[0,0].scatter(pred_comp_pca[c1,:300,0], pred_comp_pca[c1,:300,1], color="black", s=25, alpha=0.5)
ax[0,0].scatter(pred_comp_pca[c2,:300,0], pred_comp_pca[c2,:300,1], color="orange", s=25, alpha=0.5)

ax[0,0].legend(loc="upper right")
ax[0,0].grid()

ax[0,0].set_xlabel("PCA 1")
ax[0,0].set_ylabel("PCA 2")
#ax[0,0].set_ylim(0.8, 2.4)




for i in range(105):
    if i <84:
        ax[0,1].scatter(data_comp_pca[i,::2,0], data_comp_pca[i,::2,1], s=25, alpha=0.5, color="blue")
    else:
        ax[0,1].scatter(data_comp_pca[i,::2,0], data_comp_pca[i,::2,1], s=25, alpha=0.5, color="red" )
ax[0,1].scatter(trj_pca[::10,0], trj_pca[::10,1], s=25, alpha=0.5, color="purple")
ax[0,1].grid()
legend_handles = [mpatches.Patch(color="blue", label="Train set"), mpatches.Patch(color="red", label="Test set"), mpatches.Patch(color="purple", label="OOD")]
ax[0,1].legend(handles = legend_handles,loc="lower right")
ax[0,1].set_ylabel("PCA 2")
ax[0,1].set_xlabel("PCA 1") 


for i in range(106):
    if i <84:
        ax[1,0].scatter(data_comp_tsne[i,::10,0], data_comp_tsne[i,::10,1], s=25, alpha=0.5, color="blue")
    elif i == 105:
        ax[1,0].scatter(data_comp_tsne[i,::10,0], data_comp_tsne[i,::10,1], s=25, alpha=0.5, color="purple")
    else:
        ax[1,0].scatter(data_comp_tsne[i,::10,0], data_comp_tsne[i,::10,1], s=25, alpha=0.5, color="red" )

ax[1,0].grid()


ax[1,1].scatter(trj_pca[:,0], trj_pca[:,1], s=25, alpha=0.5, color="purple")
ax[1,1].scatter(pred_trj_pca[:,0],pred_trj_pca[:,1], s=25, alpha=0.5, color="grey")
ax[1,1].grid()
ax[1,1].set_ylabel("PCA 2") 
ax[1,1].set_xlabel("PCA 1")
plt.subplots_adjust(wspace=0.2, hspace=0.3)
for i, title in enumerate(["a)", "b)", "c)", "d)"]):
    
    fig.text(0.5, -0.22, title, ha='center', fontsize=12, transform=ax[int(i/2), i%2].transAxes)

fig.savefig("fig_report/latent_analysis/latent_analyis_"+args.data_type+".pdf", format="pdf")