import torch
from tqdm import tqdm   
import numpy as np


def make_sequence(data, seq_len, next_step):
    """
    Generate time-delay sequence data 
    Generates X and Y such that they do not overlap

    Args: 
        data: A numpy array follows [Ntime, Nmode] shape
        seq_len: (int) Sequence length
        next_step: (int) Number of steps to predict in the future

    Returns:
        X: Torch tensor for Input 
        Y: Torch tensor for Output
    """
    
    

    if len(data.shape) <=2:
        data    = np.expand_dims(data, axis=0)
    data = torch.tensor(data)
    nConfigs    = data.shape[0]
    nSamples    = (data.shape[1]-seq_len)
    if len(data.shape) == 4:
        X  = np.empty([nConfigs,nSamples, seq_len,data.shape[-2], data.shape[-1]])
        Y  = np.empty([nConfigs, nSamples, seq_len, data.shape[-2], data.shape[-1]])
    else:
        X  = np.empty([nConfigs,nSamples, seq_len, data.shape[-1]])
        Y  = np.empty([nConfigs, nSamples, seq_len,  data.shape[-1]])
        # Fill the input and output arrays with data
     #TODO: Adapth the output shape depending on wheter error should be computed
    # at every itermediate step or only for the prediction steps
    print("X shape:", X.shape)
    for i in tqdm(np.arange(data.shape[0])):
        k = 0
        for j in np.arange(data.shape[1]-seq_len- next_step):
                X[i,k] = data[i,j        :j+seq_len].detach().numpy()   
                #TODO: Similarly to above this would also need to be adapted
                Y[i,k] = data[i, j+next_step :j+seq_len+next_step].detach().numpy()
                k    = k + 1
    
    print(f"The training data has been generated, has shape of {X.shape, Y.shape}")

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def make_sequence2(data, sequence_length, prediction_horizon):
    batch_size = data.shape[0]

    total_length = data.shape[1]
    
    x_sequences = []
    y_sequences = []
    
    for batch in range(batch_size):
        x_batch_sequences = []
        y_batch_sequences = []
        for i in range(total_length - sequence_length - prediction_horizon+1):
            x_seq = data[batch, i:i + sequence_length, :]
            
            y_seq = data[batch, i + sequence_length:i + sequence_length + prediction_horizon, :]
            
            x_batch_sequences.append(x_seq)
            y_batch_sequences.append(y_seq)
        
        x_sequences.append(torch.stack(x_batch_sequences))
        y_sequences.append(torch.stack(y_batch_sequences))
    
    x_sequences = torch.stack(x_sequences)
    y_sequences = torch.stack(y_sequences)
    
    return x_sequences, y_sequences