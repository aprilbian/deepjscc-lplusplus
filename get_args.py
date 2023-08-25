import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', default  = 'cifar')

    # The ViT setting
    parser.add_argument('-n_patches', type=int, default  = 64)
    parser.add_argument('-n_feat', type=int, default  = 48)
    parser.add_argument('-hidden_size', type=int, default  = 256)
    parser.add_argument('-feedforward_size', type=int, default  = 1024)
    parser.add_argument('-n_layers', type=int, default  = 8)
    parser.add_argument('-dropout_prob', type=float, default  = 0.1)
    
    # The Swin setting
    parser.add_argument('-image_dims', default  = [32, 32])
    parser.add_argument('-depth', default  = [2, 4])
    parser.add_argument('-embed_size', type=int, default  = 256)
    parser.add_argument('-window_size', type=int, default  = 8)
    parser.add_argument('-mlp_ratio', type=float, default  = 4)

    parser.add_argument('-n_trans_feat', type=int, default  = 16)
    parser.add_argument('-n_heads', type=int, default  = 8)

    # The bandwidth adaption setting -> 2 types; one using different feats, one using different patches
    parser.add_argument('-min_trans_feat', type=int, default  = 1)
    parser.add_argument('-max_trans_feat', type=int, default  = 6)
    parser.add_argument('-unit_trans_feat', type=int, default  = 4)
    parser.add_argument('-trg_trans_feat', type=int, default  = 6)       # should be consistent with args.n_trans_feat
    

    parser.add_argument('-min_trans_patch', type=int, default  = 5)
    parser.add_argument('-max_trans_patch', type=int, default  = 8)
    parser.add_argument('-unit_trans_patch', type=int, default  = 8)
    parser.add_argument('-trg_trans_patch', type=int, default  = 5)       # should be consistent with args.n_trans_feat

    parser.add_argument('-n_adapt_embed', type=int, default  = 2)
    
    # channel
    parser.add_argument('-channel_mode', default = 'awgn')
    parser.add_argument('-link_qual',  default  = 7.0)
    parser.add_argument('-link_rng',  default  = 3.0)


    parser.add_argument('-adapt', default  = True)
    parser.add_argument('-full_adapt', default  = True)

    # dynamic weight adaption -- initial at 1; maximum 10; 
    parser.add_argument('-threshold', default  = 0.25)            # if it is smaller than 0.25 dB, then it's fine
    parser.add_argument('-min_clip', default  = 0)               # no smaller than 0
    parser.add_argument('-max_clip', default  = 10)              # no larger than 10
    parser.add_argument('-alpha', default  = 2)                  # weight[l] = 2**(alpha*delta[l])-1
    parser.add_argument('-freq', default  = 1)                   # The frequency of updating the weights

    # training setting
    parser.add_argument('-epoch', type=int, default  = 4000)
    parser.add_argument('-lr', type=float, default  = 1e-4)
    parser.add_argument('-train_patience', type=int, default  = 80)
    parser.add_argument('-train_batch_size', type=int, default  = 32)

    parser.add_argument('-val_batch_size', type=int, default  = 32)
    parser.add_argument('-resume', default  = False)
    parser.add_argument('-path', default  = 'models/')

    args = parser.parse_args()

    return args
