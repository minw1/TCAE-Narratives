"""
replace the exp_dir and dataset dir with your own path. 
Using 'HCP_tfMRI_Volume_4mm' to reproduce paper's result.
"""
import argparse, pathlib
import random
import torch
import numpy as np
from train_frame import Trainer, POS_Predictor_Trainer, Predictor_Tester, LM_Pretrainer_New
from data_module import Volume_Data_Module, HCP_Volume_Data_Module
from lm_datamodule import LM_Data_Module
torch.backends.cudnn.enabled = False
from randomtime_data import RT_Narrative_Data_Module

task_list = ["pieman","tunnel","lucy","prettymouth","milkywayoriginal","slumlordreach","notthefallintact","21styear","bronx","black","forgot"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fMRI Volume Embedding')

    # Models
    parser.add_argument('--model',type=str, choices=['AE','TCAE','DSRAE','DVAE','DRVAE','STAAE'], default = 'TCAE', help='Name of the embedding model')

    #Data
    parser.add_argument('--tasks', type=str, choices=['all'], default='all', help='Tasks for narrative data')
    parser.add_argument('--seg_length', type=int, default=10, help='Length of segments')
    parser.add_argument('--bold_delay', type=float, default=4.5, help='Length of BOLD delay')
    

    # Autoencoder Params
    parser.add_argument('--layers', type=int, default=1, help='Number of attention layers')
    parser.add_argument('--d_vol', type=int, default=81924, help='Number of voxels')
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


    #LM Prediction Params
    parser.add_argument('--lm_pretrain', action='store_true')

    #POS Predictor Params
    parser.add_argument('--l_vocab', type=int, default=20, help='Size of language vocabulary')
    parser.add_argument('--n_dec_blocks', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--d_dec_ff', type=int, default=512, help='Dimension of decoder feed foward layer')
    parser.add_argument('--n_dec_head', type=int, default=4, help='Number of decoder attention heads')  
    parser.add_argument('--dec_dropout', type=float, default=0.0, help='Decoder Dropout Rate')

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


    parser.add_argument('--checkpoint', type=str, help='Path to an existing checkpoint. Used along with "--resume"') #If predictor model has been training, load from here
    parser.add_argument('--encoder_base', type=str, help='Path to an existing encoder base.') 
    parser.add_argument('--decoder_base', type=str, default = "none", help='Path to an existing decoder base.') 

    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--enc_freeze_type', type=str, choices=['none','all'], default='none')
    parser.add_argument('--dec_freeze_type', type=str, choices=['none','all','thaw_cross'], default='none')
    parser.add_argument('--language_only', action='store_true')

    parser.add_argument('--perm', action='store_true')


    args = parser.parse_args()

    if args.checkpoint is not None:
        args.resume = True
    else:
        args.resume = False

    
    if str(args.exp_dir) == 'auto':
        args.exp_dir =("/home/wsm32/project/wsm_thesis_scratch/narratives/models")

    args.exp_dir = pathlib.Path(args.exp_dir)
    print('save logs to {}'.format(args.exp_dir))

    args.inference_dir = args.exp_dir/'inference'

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    if not args.lm_pretrain:
        if args.perm:
            data_module = RT_Narrative_Data_Module(task_list = task_list,
                                            batch_size = args.batch_size,
                                            num_workers = args.num_workers,
                                            delay = args.bold_delay,
                                            segment_length = args.seg_length,
                                            scramble_labels = True,
                                            scramble_seed=0)
        else:
            data_module = RT_Narrative_Data_Module(task_list = task_list,
                                                batch_size = args.batch_size,
                                                num_workers = args.num_workers,
                                                delay = args.bold_delay,
                                                segment_length = args.seg_length,
                                                scramble_labels = False
                                                )
    else:
        data_module = LM_Data_Module("/home/wsm32/project/wsm_thesis_scratch/narratives/h5_lm/wikitext_udpos_vectors", args.num_workers, args.batch_size)
                                        

    if args.test:
        policy = Predictor_Tester(args, data_module, args.exp_dir)
    elif args.lm_pretrain:
        policy = LM_Pretrainer_New(args, data_module)
    elif args.predict:
        print('Train the Predictor')
        policy = POS_Predictor_Trainer(args, data_module)
    else:
        policy = Trainer(args, data_module)

    policy()

