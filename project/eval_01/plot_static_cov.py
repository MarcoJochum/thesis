import numpy as np 
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt
batch_vec = np.linspace(1, 100, 1)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches as mpatches
from lib.data_loading import get_config
#plt.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['font.serif'] = ['Palatino']

static_cov_distr_test = np.load("eval_01/data/importance_distr_test.npy", allow_pickle=True)
static_cov_distr_train = np.load("eval_01/data/importance_distr_train.npy", allow_pickle=True)
static_cov_distr_comb= np.vstack((static_cov_distr_train, static_cov_distr_test))
# Values and names

values_test = np.mean(static_cov_distr_test, axis=0).squeeze()    
errors_test = np.std(static_cov_distr_test, axis=0).squeeze()

values_train = np.mean(static_cov_distr_train, axis=0).squeeze()
errors_train = np.std(static_cov_distr_train, axis=0).squeeze()

values_comb = np.mean(static_cov_distr_comb, axis=0).squeeze()
errors_comb = np.std(static_cov_distr_comb, axis=0).squeeze()



# ax[1].spines['top'].set_visible(False)
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['left'].set_visible(False)
# ax[1].spines['bottom'].set_visible(False)

plt.savefig("fig_report/test/stat_cov_importance.pdf", format="pdf")
   

#### Plot histograms of the relative importance of the static covariates

group_1 = static_cov_distr_train[:,0,2]> 83.0
group_2 = (static_cov_distr_train[:,0,2]< 66.0) & (static_cov_distr_train[:,0,2] > 65.0)
group_3 = (static_cov_distr_train[:,0,2]> 65.9 )& (static_cov_distr_train[:,0,2] <83.0)
group_c = (static_cov_distr_train[:,0,1]>20)#& (static_cov_distr_train[:,0,1] < 30.0)

group_1_test = static_cov_distr_test[:,0,2]> 83.0
group_2_test = (static_cov_distr_test[:,0,2]< 66.0) & (static_cov_distr_test[:,0,2] > 65.0)
group_3_test = (static_cov_distr_test[:,0,2]> 65.9 )& (static_cov_distr_test[:,0,2] <83.0)

train_config_list = get_config("../../data_kmc/2d_sets/train_set_lin_80_20_avg_list.txt")

test_config_list = get_config("../../data_kmc/2d_sets/test_set_lin_80_20_avg_list.txt")
print("group c",train_config_list[group_c,:])
print("static cov", static_cov_distr_train[group_c,0,:])    
   
config_list = np.vstack((static_cov_distr_train, static_cov_distr_test))
#group_1_comb = np.vstack((group_1.unsqueeze(), group_1_test.unsqueeze()))
#group_2_comb = np.vstack((group_2, group_2_test))
#group_3_comb = np.vstack((group_3, group_3_test))

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#list_groups = [group_1_comb, group_2_comb, group_3_comb]    
colors = ['blue', 'orange', 'red', 'green']



names = [r'$\epsilon_{r}$', r'$c_{bulk}$', r'$V_{bias}$']

# Create a bar plot

axs[0,0].bar(names, values_comb, yerr=errors_comb, align='center', alpha=0.5, ecolor='black', capsize=10)
axs[0,0].set_ylabel(r'Importance [%]')
#axs[0,0].set_title(r'Static Covariates Importance on Test Set for prediction of mean') 
axs[0,0].set_xticklabels(names, fontsize=14)  # Increase font size of x-tick labels

# ax[1].bar(names, values_train, yerr=errors_train, align='center', alpha=0.5, ecolor='black', capsize=10)
# #ax[1].set_ylabel(r'Importance [\%]')
# ax[1].set_title(r'Static Covariates Importance on train Set for prediction of mean')
# Rotate x-axis labels if needed
# Enable grid
axs[0,0].grid(True)
# ax[1].grid(True)
# Remove the frame
axs[0,0].spines['top'].set_visible(False)
axs[0,0].spines['right'].set_visible(False)
axs[0,0].spines['left'].set_visible(False)
axs[0,0].spines['bottom'].set_visible(False)


for i in range(3):
    
    
    axs[0,1].set_ylabel('Count', fontsize=12)
    if i ==2:
        n,bins, patches = axs[0,1].hist(config_list[:,0, i], bins=45, alpha=0.7, edgecolor='black')
        #axs[i].set_title(f'Distribution of relative importance of {names[i]}')
        axs[0,1].set_xlabel('Relative Importance', fontsize=12)
        for j in range(len(patches)):
                if bins[j] > 83.0:
                    patches[j].set_facecolor(colors[2])
                elif (bins[j] > 65.9)&( bins[j] < 70.0):
                    patches[j].set_facecolor(colors[1])
                elif bins[j] < 66.0:
                    patches[j].set_facecolor(colors[0])
    elif i ==0:
        n,bins, patches = axs[1,0].hist(config_list[:,0, i], bins=45, alpha=0.7, edgecolor='black')
        #axs[i].set_title(f'Distribution of relative importance of {names[i]}')
        axs[1,0].set_xlabel('Relative Importance', fontsize=12)
        for j in range(len(patches)):
            if bins[j] > 6.0:   
                patches[j].set_facecolor(colors[2])
            

    elif i ==1:
        n,bins, patches = axs[1,1].hist(config_list[:,0, i], bins=45, alpha=0.7, edgecolor='black')
        #axs[i].set_title(f'Distribution of relative importance of {names[i]}')
        axs[1,1].set_xlabel('Relative Importance', fontsize=12)
        for j in range(len(patches)):
            if bins[j] < 20.0:
                patches[j].set_facecolor(colors[2])
            elif bins[j] > 29.5:
                patches[j].set_facecolor(colors[0])
            elif (bins[j] > 20.0) & (bins[j] < 29.4):
                patches[j].set_facecolor(colors[1])
            
axs[1,0].set_ylabel('Count', fontsize=12)
axs[1,1].set_ylabel('Count', fontsize=12)
legend_handles = [
    mpatches.Patch(color=colors[0], label=r'$\varepsilon_r=1400$'),
    mpatches.Patch(color=colors[1], label=r'$\varepsilon_R=677$'),
    mpatches.Patch(color=colors[2], label=r'$\varepsilon_r=100$ ')
]
fig.text(0.5, -0.25, f"a) {names[2]}", ha='center', fontsize=14, transform=axs[0,1].transAxes)
fig.text(0.5, -0.25, f"b) {names[0]}", ha='center', fontsize=14, transform=axs[1,0].transAxes)
fig.text(0.5, -0.25, f"c) {names[1]}", ha='center', fontsize=14,transform=axs[1,1].transAxes)
fig.text(0.5, -0.25, f"d) Static covariates", ha='center', fontsize=14,transform=axs[0,0].transAxes)
axs[0,1].legend(handles=legend_handles, loc="upper center", fontsize=13)   
plt.subplots_adjust(wspace=0.2, hspace=0.5, bottom=0.2)
plt.savefig("fig_report/test/sv_histograms.pdf", format="pdf")
plt.show()


exit()
#### Histogram of encoder importance 

encoder_imp_train = np.load("eval_01/data/encoder_importance_train.npy", allow_pickle=True)
encoder_imp_test = np.load("eval_01/data/encoder_importance_test.npy", allow_pickle=True)

encoder_imp_train_mean = np.mean(encoder_imp_train, axis=0).squeeze()
encoder_imp_train_std = np.std(encoder_imp_train, axis=0).squeeze()

encoder_imp_test_mean = np.mean(encoder_imp_test, axis=0).squeeze() 
encoder_imp_test_std = np.std(encoder_imp_test, axis=0).squeeze()

#fig, ax = plt.subplots(1,1, figsize=(6, 6))
labels = ['2_target', '7_target', 'add_relative_index_futcov', '0_target', '6_target',
           '1_target', '5_target', '4_target', '9_target', '8_target','3_target']
axs[1,1].bar(labels, encoder_imp_test_mean, yerr=encoder_imp_test_std, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

fig.savefig("fig_report/test/encoder_importance_test.png", format="png")