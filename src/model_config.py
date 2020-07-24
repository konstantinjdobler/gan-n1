model_config = {}

# alpha config
model_config['alpha_jump_mode'] = "linear"
model_config['iter_alpha_jump'] = []
model_config['alpha_jump_vals'] = []
model_config['alpha_n_jumps'] = [0, 600, 600, 600, 600, 600, 600, 600, 600]
model_config['alpha_size_jumps'] = [0, 32, 32, 32, 32, 32, 32, 32, 32, 32]

# base config
model_config['max_iter_at_scale'] = [48000, 96000, 96000, 96000, 96000, 96000, 96000, 96000, 200000]
model_config['scaling_layer_channels'] = [512, 512, 512, 512, 256, 128, 64, 32, 16]
model_config['mini_batch_size'] = [16, 16, 16, 16, 16, 8, 8, 8, 8]
model_config['dim_latent_vector'] = 512
model_config['lambda_gp'] = 10
model_config["epsilon_d"] = 0.001
model_config["learning_rate"] = 0.001