import torch
from tqdm import tqdm   



def make_Sequence(data, seq_len, next_step):
    """
    Generate time-delay sequence data 

    Args: 
        data: A numpy array follows [Ntime, Nmode] shape
        seq_len: (int) Sequence length

    Returns:
        X: Numpy array for Input 
        Y: Numpy array for Output
    """
    
    

    if len(data.shape) <=2:
        data    = data.unsqueeze(0)
    #seqLen      = cfg.in_dim
    nSamples    = (data.shape[1]-seq_len)
    X           = torch.empty([nSamples, seq_len, data.shape[-1]])
    Y           = torch.empty([nSamples, next_step,data.shape[-1]])
    # Fill the input and output arrays with data
    k = 0
    for i in tqdm(torch.arange(data.shape[0])):
        for j in torch.arange(data.shape[1]-seq_len- next_step):
            X[k] = data[i, j        :j+seq_len]
            Y[k] = data[i, j+seq_len :j+seq_len+next_step]
            k    = k + 1
    print(f"The training data has been generated, has shape of {X.shape, Y.shape}")

    return X, Y