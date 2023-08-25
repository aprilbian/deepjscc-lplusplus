import numpy as np 
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS

from get_args import get_args
from swin_module_bw import *
from dataset import CIFAR10, ImageNet, Kodak
from utils import *

from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

###### Parameter Setting
args = get_args()
args.device = device

job_name = 'JSCC_swin_adapt_lr_' + args.channel_mode +'_dataset_'+str(args.dataset) + '_link_qual_' + str(args.link_qual) + '_n_trans_feat_' + str(args.n_trans_feat)\
             + '_hidden_size_' + str(args.hidden_size) + '_n_heads_' + str(args.n_heads) + '_n_layers_' + str(args.n_layers) +'_is_adapt_'+ str(args.adapt)


if args.adapt:
    job_name = job_name + '_link_rng_' + str(args.link_rng)  + '_min_trans_feat_' + str(args.min_trans_feat) + '_max_trans_feat_' + str(args.max_trans_feat) + \
                '_unit_trans_feat_' + str(args.unit_trans_feat) + '_trg_trans_feat_' + str(args.trg_trans_feat) 

print(args)
print(job_name)

writter = SummaryWriter('runs/' + job_name)

train_set = CIFAR10('datasets/cifar-10-batches-py', 'TRAIN')
valid_set = CIFAR10('datasets/cifar-10-batches-py', 'VALIDATE')
eval_set = CIFAR10('datasets/cifar-10-batches-py', 'EVALUATE')


###### The JSCC Model using Swin Transformer ######
enc_kwargs = dict(
        args = args, n_trans_feat = args.n_trans_feat, img_size=(args.image_dims[0], args.image_dims[1]),
        embed_dims=[args.embed_size, args.embed_size], depths=[args.depth[0], args.depth[1]], num_heads=[args.n_heads, args.n_heads],
        window_size=args.window_size, mlp_ratio=args.mlp_ratio, qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )

dec_kwargs = dict(
        args = args, n_trans_feat = args.n_trans_feat, img_size=(args.image_dims[0], args.image_dims[1]),
        embed_dims=[args.embed_size, args.embed_size], depths=[args.depth[1], args.depth[0]], num_heads=[args.n_heads, args.n_heads],
        window_size=args.window_size, mlp_ratio=args.mlp_ratio, norm_layer=nn.LayerNorm, patch_norm=True,)

source_enc = Swin_Encoder(**enc_kwargs).to(args.device)
source_dec = Swin_Decoder(**dec_kwargs).to(args.device)

jscc_model = Swin_JSCC(args, source_enc, source_dec)


# load pre-trained
if args.resume == False:
    pass
else:
    _ = load_weights(job_name, jscc_model)

solver = optim.Adam(jscc_model.parameters(), lr=args.lr)
scheduler = LS.MultiplicativeLR(solver, lr_lambda=lambda x: 0.9)
es = EarlyStopping(mode='min', min_delta=0, patience=args.train_patience)

###### Dataloader
train_loader = data.DataLoader(
    dataset=train_set,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=2
        )

valid_loader = data.DataLoader(
    dataset=valid_set,
    batch_size=args.val_batch_size,
    shuffle=True,
    num_workers=2
        )

eval_loader = data.DataLoader(
    dataset=eval_set,
    batch_size=args.val_batch_size,
    shuffle=True,
    num_workers=2
)

##### TARGET -> PSNR obtained by separate models
TARGET = np.array([24.75, 27.85, 30.1526917,	32.01,	33.2777652,	34.55393814])

def dynamic_weight_adaption(current):
    # dynamically change the weight; perform it every epoch
    # target -> separate trained PSNR; current -> current PSNR of adaptive model
    delta = TARGET - current
    for i in range(len(delta)):
        if delta[i] <= args.threshold:
            weight[i] = 0
        else:
            weight[i] = 2**(args.alpha*delta[i])

    clipped_weight = np.clip(weight, args.min_clip, args.max_clip)
    return clipped_weight


def train_epoch(loader, model, solvers, weight):

    model.train()

    with tqdm(loader, unit='batch') as tepoch:
        for _, (images, _) in enumerate(tepoch):
            
            epoch_postfix = OrderedDict()

            images = images.to(args.device).float()
            
            solvers.zero_grad()
            bw = np.random.randint(args.min_trans_feat, args.max_trans_feat+1)
            output = model(images, bw)

            loss = nn.MSELoss()(output, images)* weight[bw-args.min_trans_feat]
            loss.backward()
            solvers.step()

            epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())

            tepoch.set_postfix(**epoch_postfix)

def validate_epoch(loader, model, bw = args.trg_trans_feat, disable = False):

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    #msssim_hist = []
    power = []

    with torch.no_grad():
        with tqdm(loader, unit='batch', disable = disable) as tepoch:
            for _, (images, _) in enumerate(tepoch):

                epoch_postfix = OrderedDict()

                images = images.to(args.device).float()

                output = model(images, bw, snr = args.link_qual)
                
                loss = nn.MSELoss()(output, images)

                epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())

                ######  Predictions  ######
                predictions = torch.chunk(output, chunks=output.size(0), dim=0)
                target = torch.chunk(images, chunks=images.size(0), dim=0)

                ######  PSNR/SSIM/etc  ######

                psnr_vals = calc_psnr(predictions, target)
                psnr_hist.extend(psnr_vals)
                epoch_postfix['psnr'] = torch.mean(torch.tensor(psnr_vals)).item()

                ssim_vals = calc_ssim(predictions, target)
                ssim_hist.extend(ssim_vals)
                epoch_postfix['ssim'] = torch.mean(torch.tensor(ssim_vals)).item()
                
                # Show the snr/loss/psnr/ssim
                tepoch.set_postfix(**epoch_postfix)

                loss_hist.append(loss.item())
            
            loss_mean = np.nanmean(loss_hist)

            psnr_hist = torch.tensor(psnr_hist)
            psnr_mean = torch.mean(psnr_hist).item()
            psnr_std = torch.sqrt(torch.var(psnr_hist)).item()

            ssim_hist = torch.tensor(ssim_hist)
            ssim_mean = torch.mean(ssim_hist).item()
            ssim_std = torch.sqrt(torch.var(ssim_hist)).item()

            predictions = torch.cat(predictions, dim=0)[:, [2, 1, 0]]
            target = torch.cat(target, dim=0)[:, [2, 1, 0]]

            #power = torch.tensor(power)      # (num, n_patches)
            #power = power.view(-1, args.max_trans_feat)
            #power = torch.mean(power, dim=0)

            return_aux = {'psnr': psnr_mean,
                            'ssim': ssim_mean,
                            'predictions': predictions,
                            'target': target,
                            'psnr_std': psnr_std,
                            'ssim_std': ssim_std}

        
    return loss_mean, return_aux



if __name__ == '__main__':
    epoch = 0

    # initial weights
    weight = np.array([1.0 for _ in range(args.min_trans_feat, args.max_trans_feat+1)])

    while epoch < args.epoch and not args.resume:
        
        epoch += 1
        
        train_epoch(train_loader, jscc_model, solver, weight)

        valid_loss, valid_aux = validate_epoch(valid_loader, jscc_model)

        writter.add_scalar('loss', valid_loss, epoch)
        writter.add_scalar('psnr', valid_aux['psnr'], epoch)

        if epoch % args.freq == 0:
            current_psnr = np.array([0.0 for _ in range(args.min_trans_feat, args.max_trans_feat+1)])
            for i in range(len(weight)):
                _, valid_aux = validate_epoch(valid_loader, jscc_model, args.min_trans_feat + i, True)   # verbose -> True
                current_psnr[i] = valid_aux['psnr']
            
            # update the weight
            weight = dynamic_weight_adaption(current_psnr)

            ''' 
            writter.add_scalars('all_psnr', {'bw1':current_psnr[0],'bw2':current_psnr[1],'bw3':current_psnr[2],\
                                'bw4':current_psnr[3]}, epoch)
            writter.add_scalars('weights', {'weight1':weight[0],'weight2':weight[1],'weight3':weight[2],\
                                'weight4':weight[3]}, epoch)
            '''  
            writter.add_scalars('all_psnr', {'bw1':current_psnr[0],'bw2':current_psnr[1],'bw3':current_psnr[2],\
                                'bw4':current_psnr[3], 'bw5':current_psnr[4],'bw6':current_psnr[5]}, epoch)
            writter.add_scalars('weights', {'weight1':weight[0],'weight2':weight[1],'weight3':weight[2],\
                                'weight4':weight[3], 'weight5':weight[4],'weight6':weight[5]}, epoch)     


        flag, best, best_epoch, bad_epochs = es.step(torch.Tensor([valid_loss]), epoch)
        if flag:
            print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
            _ = load_weights(job_name, jscc_model)
            break
        else:
            # TODO put this in trainer
            if bad_epochs == 0:
                print('average l2_loss: ', valid_loss.item())
                save_nets(job_name, jscc_model, epoch)
                best_epoch = epoch
                print('saving best net weights...')
            elif bad_epochs % 20 == 0:
                scheduler.step()
                print('lr updated: {:.5f}'.format(scheduler.get_last_lr()[0]))



    print('evaluating...')
    print(job_name)
    #jscc_model.sr_link = 0
    ####### adjust the SNR --- fix sd_link = rd_link
    #jscc_model.sr_link = 8
    
    for tgt_trans_feat in range(1,7):
    #for link_qual in range(5,10):
        jscc_model.trg_trans_feat = tgt_trans_feat
        #args.link_qual = link_qual
        _, eval_aux = validate_epoch(eval_loader, jscc_model, bw = tgt_trans_feat)
        print(eval_aux['psnr'])
        print(eval_aux['ssim'])
        #print(eval_aux['power'])