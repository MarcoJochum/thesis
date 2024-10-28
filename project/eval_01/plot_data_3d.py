from lib.data_loading import *
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, choices=['avg', 'std'], default='avg', help='Type of data to use (avg or std)')
args = parser.parse_args()


test_data = get_config("../../data_kmc/2d_sets/test_set_lin_80_20_"+args.data_type+"_list.txt")
train_data = get_config("../../data_kmc/2d_sets/train_set_lin_80_20_"+args.data_type+"_list.txt")


##create 3d scatter plot of the train data with x axis label eprs , yaxis label c_bulk and z axis label v_bias

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_data[:,0], np.log10(train_data[:,1]), train_data[:,2], c='r', marker='o')
ax.scatter(test_data[:,0], np.log10(test_data[:,1]), test_data[:,2], c='b', marker='o')
ax.set_xlabel('EPRS')
ax.set_ylabel('C_BULK')
#sety axis to log scaling

ax.set_zlabel('V_BIAS')
ax.set_title("Train data")

fig.savefig("fig_report/test/train_data_3d.png")