"""
replace the exp_dir and dataset dir with your own path. 
Using 'HCP_tfMRI_Volume_4mm' to reproduce paper's result.
"""
import argparse, pathlib
import random
import torch
import numpy as np
from train_frame import Trainer, Predictor_Trainer, Predictor_Tester
from data_module import Volume_Data_Module, HCP_Volume_Data_Module
torch.backends.cudnn.enabled = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fMRI Volume Embedding')

    # Models
    parser.add_argument('--model',type=str, choices=['AE','TCAE','DSRAE','DVAE','DRVAE','STAAE'], default = 'TCAE', help='Name of the embedding model')

    # Dataset Params
    parser.add_argument('--dataset', type=str, choices=['HCP_tfMRI_Volume_4mm','HCP_7T_Movie_4mm','Forrest_Gump'], default='HCP_tfMRI_Volume_4mm', help='Name of the dataset')
    parser.add_argument('--task', type=str, choices=['EMOTION','LANGUAGE','MOTOR', 'WM'], default='EMOTION', help='Tasks for HCP tfMRI data')
    parser.add_argument('--mov', type=int, default=1, help='Movie series of HCP 7T movie data')
    parser.add_argument('--run', type=int, default=1, help='Run of Forrest Gump data')
    parser.add_argument('--sample_rate', type=float, default=1., help='Fraction of total samples to include')

    # Autoencoder Params
    parser.add_argument('--layers', type=int, default=1, help='Number of attention layers')
    parser.add_argument('--d_vol', type=int, default=28549, help='Number of voxels')
    parser.add_argument('--d_emd1', type=int, default=256, help='Dimension of the fisrt embedding layer')
    parser.add_argument('--d_emd2', type=int, default=128, help='Dimension of the second embedding layer')
    parser.add_argument('--d_latent', type=int, default=64, help='Dimension of latent vector ')
    parser.add_argument('--d_ff', type=int, default=512, help='Dimension of feed foward layer ')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')  
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout Rate')

    # Predictor Params
    parser.add_argument('--num_class', type=int, default=3, help='Number of classes of brain state') # EMO & LANG --3; MOTOR --7; WM --9
    parser.add_argument('--mid_planes', type=int, default=32, help='Mid-plane in the predictor')
    parser.add_argument('--pred_loss_type', type=str, choices=['mse','cross_entropy'], default='cross_entropy')
    parser.add_argument('--predict', action='store_true')

    # Training Params
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--report_period', type=int, default=50, help='Period of loss reporting')

    parser.add_argument('--loss_type', type=str, choices=['mse'], default='mse')  

    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--lr_step_size', type=int, default=30, help='Period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=0., help='Strength of weight decay regularization')
    parser.add_argument('--early_stop', type=int, default=5, help='waiting epochs')

    # Device
    parser.add_argument('--num_workers', type=int, default=2, help='Number of CPU workers to load the data')    
    parser.add_argument('--device', type=str, default='cuda', help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--data_parallel', action='store_true',help='If set, use multiple GPUs using data parallelism')

    # Saving and Loggings
    parser.add_argument('--exp_dir', type=pathlib.Path, required=True, help='Path where model and results should be saved')
    parser.add_argument('--save_model', type=bool, default=True, help='Save model every iteration or not')
    parser.add_argument('--save_embedding', type=bool, default=True, help='Save embedding in test or not')
    parser.add_argument('--checkpoint', type=str, help='Path to an existing checkpoint. Used along with "--resume"')

    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')

    parser.add_argument('--test', action='store_true')


    args = parser.parse_args()

    if args.checkpoint is not None:
        args.resume = True
    else:
        args.resume = False

    
    if str(args.exp_dir) == 'auto' and args.model == 'TCAE' and args.dataset == 'HCP_tfMRI_Volume_4mm':
        print('sec')
        args.exp_dir =('/home/zihao/Projects/code/checkpoints/main/'+args.task+'_'+args.model+
             '_d_latent='+str(args.d_latent)+'_'+str(args.layers)+'_'+str(args.n_head)+
             '_epochs=' + str(args.num_epochs) +'_lr=' + str(args.lr)
              )

    elif str(args.exp_dir) == 'auto' and args.dataset == 'HCP_tfMRI_Volume_4mm':
        args.exp_dir =('/home/zihao/Projects/code/checkpoints/baseline/'+args.task+'_'+args.model+
             '_d_latent='+str(args.d_emd1)+'_'+str(args.d_emd2)+'_'+str(args.d_latent)+
             '_epochs=' + str(args.num_epochs) +'_lr=' + str(args.lr)
              )

    elif str(args.exp_dir) == 'auto'and args.dataset == 'HCP_7T_Movie_4mm':
        args.exp_dir =('/home/zihao/Projects/code/checkpoints/tcae_trans/'+args.dataset +'_MOV'+ str(args.mov) + '_'+args.model+'_d_latent='+str(args.d_latent)+
             '_epochs=' + str(args.num_epochs) +'_bs=' + str(args.batch_size)+
             '_loss_type='+args.loss_type +'_lr=' + str(args.lr)
              )        

    elif str(args.exp_dir) == 'auto'and args.dataset == 'Forrest_Gump':
        args.exp_dir =('/home/zihao/Projects/code/checkpoints/'+args.dataset +'_run'+ str(0) + '_'+args.model+'_d_latent='+str(args.d_latent)+
             '_epochs=' + str(args.num_epochs) +'_bs=' + str(args.batch_size)+
             '_loss_type='+args.loss_type +'_lr=' + str(args.lr)
              )        

    args.exp_dir = pathlib.Path(args.exp_dir)
    print('save logs to {}'.format(args.exp_dir))

    args.inference_dir = args.exp_dir/'inference'

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)


    if args.dataset == 'HCP_7T_Movie_4mm':
        args.data_path = "/home/zihao/nvme/data/HCP_Movie_Volume_4mm/data_split_new/MOV"+str(args.mov)+"/"
        data_module = Volume_Data_Module(
            data_path = args.data_path,
            batch_size = args.batch_size,    
            sample_rate= args.sample_rate,
            )
    elif args.dataset == 'Forrest_Gump':
        args.data_path = "/home/zihao/nvme/data/Forrest_Gump/data_split/run"+str(args.run)+"/"
        print("args run",args.run)
        data_module = Volume_Data_Module(
            data_path = args.data_path,
            batch_size = args.batch_size,    
            sample_rate= args.sample_rate,
            )
    elif args.dataset == 'HCP_tfMRI_Volume_4mm':
        args.data_path = "/home/zihao/nvme/data/HCP_tfMRI_Volume_4mm/"
        data_module = HCP_Volume_Data_Module(
            data_path = args.data_path,
            task = args.task,
            batch_size = args.batch_size,
            sample_rate = args.sample_rate,
            num_workers = args.num_workers
        )
    else:
        raise NotImplementedError()

    if args.test:
        print
        policy = Predictor_Tester(args, data_module, args.exp_dir)
    elif args.predict:
        print('Train the Predictor')
        policy = Predictor_Trainer(args, data_module)
    else:
        policy = Trainer(args, data_module)

    policy()

