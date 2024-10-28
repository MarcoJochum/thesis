

from .vae import VAE_config
class LSTM_config:

    ''' Configuration for the LSTM model '''    
    batch_size = 200

    latent_dim = VAE_config.latent_dim
    base = VAE_config.base
    part_class = VAE_config.part_class


    n_epochs = 100
    lr = 0.001
    hidden_size = 50
    num_layers = 1
    num_classes = 1
    seq_len = 10

    data_train = VAE_config.data_train
    data_test = VAE_config.data_test

    train_params = VAE_config.train_params
    test_params = VAE_config.test_params

    model_name = 'model_lstm_lin_vae.pth'
    vae_name = VAE_config.model_name  