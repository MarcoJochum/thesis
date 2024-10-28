
import matplotlib.pyplot as plt

from lib.helper import *
import matplotlib.patches as mpatches   

import matplotlib.lines as mlines
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

trj_2_pca= np.load("eval_01/data/trj_2_pca_"+args.data_type+".npy")
pred_trj_2_pca = np.load("eval_01/data/y_pred_trj_2_pca_"+args.data_type+".npy")
trj_15_pca= np.load("eval_01/data/trj_15_pca_"+args.data_type+".npy")
pred_trj_15_pca = np.load("eval_01/data/y_pred_trj_15_pca_"+args.data_type+".npy")
trj_1_pca= np.load("eval_01/data/trj_1_pca_"+args.data_type+".npy")
pred_trj_1_pca = np.load("eval_01/data/y_pred_trj_1_pca_"+args.data_type+".npy")

if args.data_type == 'avg':
    pred_comp_pca = np.reshape(pred_comp_pca, (105, 1000, 2))
else:
    pred_comp_pca = np.reshape(pred_comp_pca, (105, 995, 2))
fig, ax = plt.subplots(1, 3, figsize=(15, 5))


## Scatter plot for two configs
c1 = 1
c2 = 87    
c3= 17

ax[2].scatter(data_comp_pca[c1,:,0], data_comp_pca[c1,:,1], color="black", s=25, alpha=0.5,label=f"Ground truth $\Lambda^1$") 
ax[2].scatter(data_comp_pca[c2,:,0], data_comp_pca[c2,:,1], color="brown", s=25, alpha=0.5, label=f'Ground truth $\Lambda^2$')
k=1
for i, (x, y) in enumerate(zip(data_comp_pca[c2,:,0], data_comp_pca[c2,:,1])):
    if k <11:
        ax[2].annotate(f'{k}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black')
    
    k+=1
    if k==500:
        k=0

k=1
if args.data_type == 'std':
    for i, (x, y) in enumerate(zip(data_comp_pca[c1,:,0], data_comp_pca[c1,:,1])):
        if k <11:
            ax[2].annotate(f'{k}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black')
    
        k+=1
        if k==500:
            k=0
#Prediction data

ax[2].scatter(pred_comp_pca[c1,:300,0], pred_comp_pca[c1,:300,1], color="grey", s=25, alpha=0.5, label="Prediction $\Lambda^1$")
ax[2].scatter(pred_comp_pca[c2,:300,0], pred_comp_pca[c2,:300,1], color="orange", s=25, alpha=0.5, label="Prediction $\Lambda^2$")


ax[2].grid()
if args.data_type == 'std':
    ax[2].set_ylim(-5, 17.5)   
    ax[2].set_xlim(0.0, 7)
ax[2].set_xlabel("PC 1")
ax[2].set_ylabel("PC 2")
#ax[2].set_ylim(0.8, 2.4)

if args.data_type == 'std':
    ax[2].legend(loc="lower right",  borderaxespad=0.1) #bbox_to_anchor=(1.05, 0.5),
else:
    ax[2].legend(loc="lower left",  borderaxespad=0.1)
for i in range(105):
    if i <84:
        ax[0].scatter(data_comp_pca[i,::2,0], data_comp_pca[i,::2,1], s=25, alpha=0.5, color="blue")
    else:
        ax[0].scatter(data_comp_pca[i,::2,0], data_comp_pca[i,::2,1], s=25, alpha=0.5, color="red" )
ax[0].scatter(trj_2_pca[::10,0], trj_2_pca[::10,1], s=25, alpha=0.5, color="grey")
ax[0].scatter(trj_15_pca[::10,0], trj_15_pca[::10,1], s=25, alpha=0.5, color="grey")
ax[0].scatter(trj_1_pca[::10,0], trj_1_pca[::10,1], s=25, alpha=0.5, color="grey")

ax[0].grid()

legend_handles = [mlines.Line2D([], [], alpha=0.5, color='blue', marker='o', linestyle='None', markersize=10, label='Train set'),
                   mlines.Line2D([], [], alpha=0.5, color='red', marker='o', linestyle='None', markersize=10, label='Test set'),
                   mlines.Line2D([], [], alpha=0.5, color='grey', marker='o', linestyle='None', markersize=10, label=r'High $c_{bulk}$')]
ax[0].legend(handles = legend_handles,loc="lower right",  borderaxespad=0.1)
ax[0].set_ylabel("PC 2")
ax[0].set_xlabel("PC 1") 
if args.data_type == 'std':
    ax[0].set_xlim(-100, 150)



for i in range(108):
    if i <84:
        ax[1].scatter(data_comp_tsne[i,::10,0], data_comp_tsne[i,::10,1], s=25, alpha=0.5, color="blue")
    elif i > 104:
        ax[1].scatter(data_comp_tsne[i,::10,0], data_comp_tsne[i,::10,1], s=25, alpha=0.5, color="grey")
    else:
        ax[1].scatter(data_comp_tsne[i,::10,0], data_comp_tsne[i,::10,1], s=25, alpha=0.5, color="red" )

ax[1].grid()
ax[1].set_ylabel("t-SNE 2")
ax[1].set_xlabel("t-SNE 1")

# ax[3].scatter(trj_2_pca[:,0], trj_2_pca[:,1], s=25, alpha=0.5, color="deepskyblue")
# ax[3].scatter(pred_trj_2_pca[:,0],pred_trj_2_pca[:,1], s=25, alpha=0.5,marker="x", color="deepskyblue")
# ax[3].scatter(trj_15_pca[:,0], trj_15_pca[:,1], s=25, alpha=0.5, color="skyblue")
# ax[3].scatter(pred_trj_15_pca[:,0],pred_trj_15_pca[:,1], s=25, alpha=0.5,marker="x", color="skyblue")
# ax[3].scatter(trj_1_pca[:,0], trj_1_pca[:,1], s=25, alpha=0.5, color="steelblue")
# ax[3].scatter(pred_trj_1_pca[:,0],pred_trj_1_pca[:,1], s=25, alpha=0.5, marker="x",color="steelblue")
# ax[3].grid()
# ax[3].set_ylabel("PC 2") 
# ax[3].set_xlabel("PC 1")


# marker_legend_handles = [
#     mlines.Line2D([], [], alpha=0.5, color='grey', marker='o', linestyle='None', markersize=10, label='Ground truth'),
#     mlines.Line2D([], [], color='grey', marker='x', linestyle='None', markersize=10, label='Prediction')
# ]
# ax[3].legend(handles=marker_legend_handles, loc='lower right',  borderaxespad=0.1)


for i, title in enumerate([f"a) PCA of the data set", "b) t-SNE of the data set", "c) PCA: Prediction and ground truth"]):
    
    fig.text(0.5, -0.22, title, ha='center', fontsize=12, transform=ax[i].transAxes)  #[int(i/2), i%2]
plt.subplots_adjust(wspace=0.4, hspace=0.3)
fig.savefig("fig_report/latent_analysis/latent_analyis_"+args.data_type+"_presentation.png", format="png",bbox_inches='tight', dpi=600) 

print(f"$\Lambda^1=$ [{config_list[c1,0]:.0f}, {config_list[c1,1]:.1e}, {config_list[c1,2]:.2}], $\Lambda^2=$[{config_list[c2,0]:.0f}, {config_list[c2,1]:.1e}, {config_list[c2,2]:.2} ]")