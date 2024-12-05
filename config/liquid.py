from .vae import VAE_config

class Liquid_config:

    ''' Configuration for the liquid NN model '''

    n_epochs = 100
    batch_size = 2000
    latent_dim = VAE_config.latent_dim
    base = VAE_config.base
    part_class = VAE_config.part_class


    lr = 0.001
    units = 100

    proj_size = latent_dim
    backbone_layers = 3
    backbone_units = 20
    backbone_dropout = 0.1
    activation = "tanh" # "relu", "silu", "gelu"
    data_train = VAE_config.data_train
    data_test = VAE_config.data_test
    
    train_params = VAE_config.train_params
    test_params = VAE_config.test_params

    model_name = 'no_name_specified.pth'
    vae_name = VAE_config.model_name  