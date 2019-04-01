from __future__ import print_function
from __future__ import division

import nmt.all_constants as ac


def bpe_en2es():
    config = {}

    config['model_name']        = 'en2es'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'es'
    config['data_dir']          = './nmt/data/en2es'
    config['checkpoints']       = ['clean', 'article', 'dropone', 'nounnum', 'prep', 'sva']
    config['log_file']          = './nmt/DEBUG.log'
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8
    config['learned_pos']       = False
    config['max_pos_length']    = 1024 # don't ever let me go further than this pls
    config['dropout']           = 0.2
    config['batch_size']        = 4096
    config['init_type']         = ac.XAVIER_NORMAL
    config['max_epochs']        = 50
    config['validate_freq']     = 5000
    config['val_per_epoch']     = False # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['max_length']        = 256
    config['length_ratio']      = 3.0
    config['label_smoothing']   = 0.1
    config['normalize_loss']    = ac.LOSS_TOK # don't see any difference between loss_batch and loss_tok
    # if use adam
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.ORG_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['min_lr']            = 1e-5
    config['patience']          = 3
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['grad_clip']         = 5.0
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = True
    config['beam_size']         = 4
    config['beam_alpha']        = 0.6

    return config
