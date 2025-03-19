import os,time
import shutil, pickle, logging
import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from baselines import Baseline_Autoencoder, vae_loss_function
from model import Make_Autoencoder,subsequent_mask
from predictor import Brain_State_Predictor
from utils import AverageMeter,accuracy
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from torchmetrics.functional.text import perplexity
from torchmetrics.functional.classification import multiclass_accuracy
from pos_decoder import POS_predictor, No_Scans_POS_Decoder, POS_Decoder
from utils import pos_tags

vae_loss = vae_loss_function()

def build_optimizer(lr, weight_decay, model_parameters, type='adam'):
    if type == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_parameters), lr, weight_decay=weight_decay)
    elif type == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_parameters), lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError()

    return optimizer


def build_lr_scheduler(options, optimizer, last_epoch=-1):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, options.lr_step_size, options.lr_gamma, last_epoch=last_epoch)
    return scheduler


def build_model(options):

    # 1 embed
    if options.model == 'TCAE':
        model = Make_Autoencoder(
            N=options.layers, 
            d_vol=options.d_vol,
            d_latent=options.d_latent, 
            d_ff=options.d_ff, 
            h=options.n_head, 
            dropout=options.dropout).to(options.device)
    else:
        raise NotImplementedError()

    return model


def build_predictor(options):

    model = Brain_State_Predictor(
        num_class=options.num_class,
        in_planes=options.d_latent,
        mid_planes=options.mid_planes
        ).to(options.device)

    return model

def build_pos_predictor(options):

    if options.language_only or options.lm_pretrain:
        model = POS_Decoder(options.l_vocab, d_latent=options.d_latent, d_ff=options.d_dec_ff, dropout=options.dec_dropout, n_head=options.n_dec_head, n_blocks=options.n_dec_blocks)
    else:
        model = POS_predictor(options.l_vocab, d_latent=options.d_latent, d_dec_ff=options.d_dec_ff, dec_dropout=options.dec_dropout, n_dec_head=options.n_dec_head, n_dec_blocks=options.n_dec_blocks, d_vol=options.d_vol, d_enc_ff=options.d_ff,n_enc_head=options.n_head,n_enc_blocks=options.layers,enc_dropout=options.dropout)
    return model

#def build_pos_decoder(options):
#    model = POS_Decoder(options.l_vocab, d_latent=options.d_latent, d_ff=options.d_dec_ff, dropout=options.dec_dropout, n_head=options.n_dec_head, n_blocks=options.n_dec_blocks)
#    return model

class Trainer:
    """
    Trainer.

    Configuration for the trainer is provided by the argument 'options'. 
    Must contain the following fields:

    Args:
        options('argparse.Namespace'): Options for the trainer.
        data_module('object'): Data module class for the used datasets.
    """
    def __init__(self, options, data_module):
        self.options = options

        # setup model, optimizer and scheduler
        self.model = build_model(self.options)
        if options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.model.device_ids = [0,1]
        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.scheduler = build_lr_scheduler(self.options, self.optimizer)

        # setup the dataloader
        self.train_loader, self.val_loader, _ = data_module._setup_dataloaders()

        self.epoch = 0
        self.start_epoch = 0
        self.end_epoch = options.num_epochs

        self.best_val_loss = np.inf
        self.best_epoch = 0

        # setup saving, writer, and logging
        options.exp_dir.mkdir(parents=True, exist_ok=True)
        options.inference_dir.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(str(options.exp_dir), 'args.pkl'), "wb") as f:
            pickle.dump(options.__dict__, f)

        self.writer = SummaryWriter(log_dir=options.exp_dir / 'summary')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)
        self.logger.info(self.model)

    def load_checkpoint(self):
        if self.options.resume:
            self.load()

    def __call__(self):
        self.load_checkpoint()
        return self.train()

    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            train_loss, train_time = self.train_epoch(self.epoch)
            self.scheduler.step()
            val_loss, val_time = self.evaluate_epoch()

            is_best = val_loss < self.best_val_loss
            self.best_val_loss = min(self.best_val_loss, val_loss)
            if self.options.save_model:
                self.save_model(is_best)
            self.logger.info(
                f'Epoch = [{1 + self.epoch:3d}/{self.options.num_epochs:3d}] Train_Loss = {train_loss:.4g} '
                f'Val_Loss = {val_loss:.4g} Train_Time = {train_time:.4f}s Val_Time = {val_time:.4f}s',
            )
            if is_best:
                self.best_epoch = epoch
            print('Best epoch: ',self.best_epoch)
            print('Waiting: ', epoch-self.best_epoch)
            if (epoch-self.best_epoch) >= self.options.early_stop:
                print("######### Early Stop #########")
                break
        self.writer.close()

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = AverageMeter()
        batch_time = AverageMeter()
        start_epoch = time.perf_counter()

        for iter, data in enumerate(self.train_loader):
            start_iter = time.perf_counter()

            #print(data[0].to(self.options.device).type())
            signal = data[0].to(self.options.device).float()

            # tgt_mask = self.make_std_mask(signal, 0)
            #print(tgt_mask.shape)

            if self.options.model == 'DVAE' or self.options.model == 'DRVAE':
                output, mu, log_var = self.model(signal)
                loss = vae_loss(output, signal, mu, log_var)[0]
                recon_loss = self.loss(output, signal, self.options.loss_type)
                total_loss.update(recon_loss.item())
            
            elif self.options.model == 'TCAE':
                # output = self.model(signal, None, tgt_mask) # mask
                output = self.model(signal, None, None)
                # print('*******__batch__***********')
                loss = self.loss(output, signal, self.options.loss_type)
                total_loss.update(loss.item())

            else:
                output = self.model(signal)
                loss = self.loss(output, signal, self.options.loss_type)
                total_loss.update(loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if iter % self.options.report_period == 0:
                print('Epoch[{0}][{1}/{2}]\t'
                    'Time {batch_time.avg:.3f} ({batch_time.val:.3f})\t' 
                    'Loss {total_loss.avg:.4f} ({total_loss.val:.4f})'.format(
                    epoch, iter, len(self.train_loader), batch_time = batch_time, total_loss = total_loss))

            batch_time.update(time.perf_counter()-start_iter)
               
        self.writer.add_scalar('Train_Loss', total_loss.avg, self.epoch)
        print('Train Time {0} \t Train Loss {1}'.format(time.perf_counter() - start_epoch, total_loss.avg))

        return total_loss.avg, time.perf_counter() - start_epoch

    def evaluate_epoch(self):
        self.model.eval()

        total_loss = AverageMeter()
        start_epoch = time.perf_counter()

        with torch.no_grad():
            for iter, data in enumerate(self.val_loader):
                start_iter = time.perf_counter()
                signal = data[0].to(self.options.device).float()

                # tgt_mask = self.make_std_mask(signal, 0)

                if self.options.model == 'DVAE' or self.options.model == 'DRVAE':
                    output, mu, log_var = self.model(signal)
                    loss = vae_loss(output, signal, mu, log_var)[0]
                    mse_loss = self.loss(output, signal, self.options.loss_type)

                    total_loss.update(mse_loss.item())
                
                elif self.options.model == 'TCAE':
                    # output = self.model(signal, None, tgt_mask) # mask
                    output = self.model(signal, None, None)
                    loss = self.loss(output, signal, self.options.loss_type)
                    total_loss.update(loss.item())
                else:
                    output = self.model(signal)
                    loss = self.loss(output, signal, self.options.loss_type)
                    total_loss.update(loss.item())

            self.writer.add_scalar('Val_Loss', total_loss.avg, self.epoch)
            print('Val Time {0} \t Val Loss {1}'.format(time.perf_counter() - start_epoch, total_loss.avg))

        return total_loss.avg, time.perf_counter() - start_epoch

    def loss(self, input, target, loss_type):
        """
        Args:
            input: reconstructed time series
            target: original time series
            loss_type: the type of loss function         
        """
        # Reconstruction Loss
        if loss_type == 'mse':
            loss = F.mse_loss(input,target)
        else:
            raise NotImplementedError()

        return loss

    def load(self):
        self.model = build_model(self.options)
        if self.options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        checkpoint_tmp = torch.load(self.options.checkpoint)
        print("Load checkpoint {} with loss {}".format(checkpoint_tmp['epoch'], checkpoint_tmp['best_val_loss']))
        self.best_val_loss = checkpoint_tmp['best_val_loss']
        self.best_epoch = checkpoint_tmp.get('best_epoch', 0)
        self.start_epoch = checkpoint_tmp['epoch'] + 1

        self.model.load_state_dict(checkpoint_tmp['model'])
        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.optimizer.load_state_dict(checkpoint_tmp['optimizer'])
        self.scheduler = build_lr_scheduler(self.options, self.optimizer,self.start_epoch)

    def save_model(self,is_best):
        exp_dir = self.options.exp_dir
        torch.save(
            {
                'epoch': self.epoch,
                'options': self.options,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'exp_dir': exp_dir,
                'best_epoch': self.best_epoch,  # Save the best epoch
            },
            f = exp_dir / 'model.pt'
        )
        if is_best:
            shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


class Tester:
    def __init__(self, options, data_module, exp_dir):

        # setup the dataloader
        _, _, self.test_loader = data_module._setup_dataloaders()

        # load options and model
        self.load(os.path.join(exp_dir , 'best_model.pt'))
        self.options.exp_dir = exp_dir
        self.options.inference_dir = options.inference_dir  

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)
        self.logger.info(self.model)

    def __call__(self):
        self.evaluate()

    def evaluate(self):
        self.model.eval()

        top1 = AverageMeter()
        
        Label = []
        Pred = []

        with torch.no_grad():
            for iter, data in enumerate(self.test_loader):

                nodes = data[0].detach().cpu().numpy().tolist()
                labels = data[1].to(self.options.device)
                coor = data[2].to(self.options.device).float()
                struct_feat = data[3].to(self.options.device).float()

                output = self.model(nodes, adj_lists, features, coor,struct_feat)

                accu = accuracy(output,labels,[1,])
                top1.update(accu[0].item())

                Label.append(labels.detach().cpu().numpy())

                _, pred = output.topk(1, 1, True, True)
                Pred.append(pred.detach().cpu().numpy())

            Label = np.concatenate(Label,axis=0)
            Pred = np.concatenate(Pred,axis=0).squeeze()

            np.save(self.options.inference_dir/'pred.npy',Pred)
            np.save(self.options.inference_dir/'label.npy',Label)

            print('Accuracy: {0:.3f}'.format(top1.avg))


    def load(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        print("Load checkpoint {} with loss {}".format(checkpoint['epoch'], checkpoint['best_val_loss']))
        self.options = checkpoint['options']

        self.model = build_model(self.options)
        if self.options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.model.device_ids = [0,1]

        self.model.eval()

        self.model.load_state_dict(checkpoint['model'])

    def save(self, fname, array):
        np.save(fname, array)

class Predictor_Trainer:
    """
    Trainer.

    Configuration for the predictor trainer is provided by the argument 'options'. 
    Must contain the following fields:

    Args:
        options('argparse.Namespace'): Options for the trainer.
        data_module('object'): Data module class for the used datasets.
    """
    def __init__(self, options, data_module):
        self.options = options

        # setup model, optimizer and scheduler
        self.model = build_predictor(self.options)
        if options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.model.device_ids = [0,1]
        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.scheduler = build_lr_scheduler(self.options, self.optimizer)

        # setup the dataloader
        self.train_loader, self.val_loader, _ = data_module._setup_dataloaders()

        self.epoch = 0
        self.start_epoch = 0
        self.end_epoch = options.num_epochs

        self.best_val_loss = np.inf
        self.best_epoch = 0

        # setup saving, writer, and logging
        options.exp_dir.mkdir(parents=True, exist_ok=True)
        options.inference_dir.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(str(options.exp_dir), 'args.pkl'), "wb") as f:
            pickle.dump(options.__dict__, f)

        self.writer = SummaryWriter(log_dir=options.exp_dir / 'summary')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)
        self.logger.info(self.model)

    def load_checkpoint(self):
        if self.options.resume:
            self.load()

    def __call__(self):
        self.load_checkpoint()
        self.load_embed(os.path.join(self.options.exp_dir , 'best_model.pt'))
        return self.train()

    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            train_loss, train_time = self.train_epoch(self.epoch)
            self.scheduler.step()
            val_loss, val_time = self.evaluate_epoch()

            is_best = val_loss < self.best_val_loss
            self.best_val_loss = min(self.best_val_loss, val_loss)
            if self.options.save_model:
                self.save_model(is_best)
            self.logger.info(
                f'Epoch = [{1 + self.epoch:3d}/{self.options.num_epochs:3d}] Train_Loss = {train_loss:.4g} '
                f'Val_Loss = {val_loss:.4g} Train_Time = {train_time:.4f}s Val_Time = {val_time:.4f}s',
            )
            if is_best:
                self.best_epoch = epoch
            print('Best epoch: ',self.best_epoch)
            print('Waiting: ', epoch-self.best_epoch)
            if (epoch-self.best_epoch) >= self.options.early_stop:
                print("######### Early Stop #########")
                break
        self.writer.close()

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = AverageMeter()
        batch_time = AverageMeter()
        pred_acc = AverageMeter()
        start_epoch = time.perf_counter()

        for iter, data in enumerate(self.train_loader):
            start_iter = time.perf_counter()
            signal = data[0].to(self.options.device).float()
            label = data[1].to(self.options.device).long()
            

            if self.options.model == 'DVAE' or self.options.model == 'DRVAE':
                embed, mu, log_var = self.embed.encode(signal)
                output = self.model(embed)
                loss = self.loss(output.reshape(-1,self.options.num_class), label.reshape(-1), self.options.pred_loss_type)
                total_loss.update(loss.item())
            
            elif self.options.model == 'TCAE':
                embed = self.embed.encode(signal, None)             
                output = self.model(embed)
                loss = self.loss(output.reshape(-1,self.options.num_class), label.reshape(-1), self.options.pred_loss_type)
                total_loss.update(loss.item())
            else:
                embed = self.embed.encode(signal)
                output = self.model(embed)
                loss = self.loss(output.reshape(-1,self.options.num_class), label.reshape(-1), self.options.pred_loss_type)
                total_loss.update(loss.item())

            acc = accuracy(output.reshape(-1,self.options.num_class),label.reshape(-1))
            pred_acc.update(acc[0].item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if iter % self.options.report_period == 0:
                print('Epoch[{0}][{1}/{2}]\t'
                    'Time {batch_time.avg:.3f} ({batch_time.val:.3f})\t' 
                    'Loss {total_loss.avg:.4f} ({total_loss.val:.4f})\t'
                    'Acc {pred_acc.avg:.4f} ({pred_acc.val:.4f})'.format(
                    epoch, iter, len(self.train_loader), batch_time = batch_time, total_loss = total_loss, pred_acc = pred_acc))

            batch_time.update(time.perf_counter()-start_iter)
               
        self.writer.add_scalar('Train_Loss', total_loss.avg, self.epoch)
        print('Train Time {0} \t Train Loss {1}'.format(time.perf_counter() - start_epoch, total_loss.avg))

        return total_loss.avg, time.perf_counter() - start_epoch

    def evaluate_epoch(self):
        self.model.eval()

        total_loss = AverageMeter()
        pred_acc = AverageMeter()
        start_epoch = time.perf_counter()

        with torch.no_grad():
            for iter, data in enumerate(self.val_loader):
                signal = data[0].to(self.options.device).float()               
                label = data[1].to(self.options.device).long()

                if self.options.model == 'DVAE' or self.options.model == 'DRVAE':
                    embed, mu, log_var = self.embed.encode(signal)
                    output = self.model(embed)
                    loss = self.loss(output.reshape(-1,self.options.num_class), label.reshape(-1), self.options.pred_loss_type)
                    total_loss.update(loss.item())
                
                elif self.options.model == 'TCAE':
                    embed = self.embed.encode(signal,None)
                    output = self.model(embed)
                    loss = self.loss(output.reshape(-1,self.options.num_class), label.reshape(-1), self.options.pred_loss_type)
                    total_loss.update(loss.item())
                else:
                    embed = self.embed.encode(signal)
                    output = self.model(embed)
                    loss = self.loss(output.reshape(-1,self.options.num_class), label.reshape(-1), self.options.pred_loss_type)
                    total_loss.update(loss.item())

            acc = accuracy(output.reshape(-1,self.options.num_class),label.reshape(-1))
            pred_acc.update(acc[0].item())

            self.writer.add_scalar('Val_Loss', total_loss.avg, self.epoch)
            print('Val Time {0} \t Val Loss {1} \t Val Acc {2}'.format(time.perf_counter() - start_epoch, total_loss.avg, pred_acc.avg))

        return total_loss.avg, time.perf_counter() - start_epoch

    def loss(self, input, target, loss_type):
        """
        Args:
            input: reconstructed time series
            target: original time series
            loss_type: the type of loss function         
        """
        # Reconstruction Loss
        if loss_type == 'cross_entropy':
            # print('input: ',input.shape)
            # print('target: ',np.unique(target.detach().cpu().numpy()))
            loss = F.cross_entropy(input,target)
        elif loss_type == 'mse':
            loss = F.mse_loss(input,target)
        else:
            raise NotImplementedError()

        return loss

    def load_embed(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        print("Load checkpoint {} with loss {}".format(checkpoint['epoch'], checkpoint['best_val_loss']))
        #self.options = checkpoint['options']

        self.embed = build_model(self.options)
        if self.options.data_parallel:
            self.embed = torch.nn.DataParallel(self.model)
            self.embed.device_ids = [0,1]

        self.embed.eval()

        self.embed.load_state_dict(checkpoint['model'])

    def load(self):
        self.model = build_model(self.options)
        if self.options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        checkpoint_tmp = torch.load(self.options.checkpoint)
        print("Load checkpoint {} with loss {}".format(checkpoint_tmp['epoch'], checkpoint_tmp['best_val_loss']))
        self.best_val_loss = checkpoint_tmp['best_val_loss']
        self.start_epoch = checkpoint_tmp['epoch'] + 1

        self.model.load_state_dict(checkpoint_tmp['model'])
        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.optimizer.load_state_dict(checkpoint_tmp['optimizer'])
        self.scheduler = build_lr_scheduler(self.options, self.optimizer,self.start_epoch)


    def save_model(self,is_best):
        exp_dir = self.options.exp_dir
        torch.save(
            {
                'epoch': self.epoch,
                'options': self.options,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'exp_dir': exp_dir
            },
            f = exp_dir / 'predictor_model.pt'
        )
        if is_best:
            shutil.copyfile(exp_dir / 'predictor_model.pt', exp_dir / 'best_predictor_model.pt')

class POS_Predictor_Trainer:
    """
    Trainer.

    Configuration for the POS predictor trainer is provided by the argument 'options'. 
    Must contain the following fields:

    Args:
        options('argparse.Namespace'): Options for the trainer.
        data_module('object'): Data module class for the used datasets.
    """
    def __init__(self, options, data_module):
        self.options = options

        # setup model, optimizer and scheduler
        self.model = build_pos_predictor(self.options)
        self.model.initialize_encoder_weights(self.options.encoder_base)
        self.model.initialize_decoder_weights(self.options.decoder_base)
        self.model = self.model.to(self.options.device)

        if options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.model.device_ids = [0,1]
        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.scheduler = build_lr_scheduler(self.options, self.optimizer)
        

        # setup the dataloader
        self.train_loader, self.val_loader, _ = data_module._setup_dataloaders()

        self.epoch = 0
        self.start_epoch = 0
        self.end_epoch = options.num_epochs

        self.best_val_loss = np.inf
        self.best_epoch = 0

        # setup saving, writer, and logging
        options.exp_dir.mkdir(parents=True, exist_ok=True)
        options.inference_dir.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(str(options.exp_dir), 'args.pkl'), "wb") as f:
            pickle.dump(options.__dict__, f)

        self.writer = SummaryWriter(log_dir=options.exp_dir / 'summary')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)
        self.logger.info(self.model)

    def load_checkpoint(self):
        if self.options.resume:
            self.load()

    def __call__(self):
        self.load_checkpoint()
        return self.train()

    def train(self):
        #once before any training
        eval_res = self.evaluate(self.val_loader)
        val_loss = eval_res["loss"]
        val_perp = eval_res["perp"]
        val_time = eval_res ["time"]

        eval_tres = self.evaluate(self.train_loader)
        train_loss = eval_tres["loss"]
        train_perp = eval_tres["perp"]
        train_time = eval_tres ["time"]

        self.writer.add_scalar('Train_Loss', train_loss, self.epoch)
        self.writer.add_scalar('Val_Loss', val_loss, self.epoch)

        self.writer.add_scalar('Train_Perp', train_perp, self.epoch)
        self.writer.add_scalar('Val_Perp', val_perp, self.epoch)

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train_epoch(self.epoch)
            self.scheduler.step()
            eval_res = self.evaluate(self.val_loader)
            val_loss = eval_res["loss"]
            val_perp = eval_res["perp"]
            val_time = eval_res ["time"]

            eval_tres = self.evaluate(self.train_loader)
            train_loss = eval_tres["loss"]
            train_perp = eval_tres["perp"]
            train_time = eval_tres ["time"]

            self.writer.add_scalar('Train_Loss', train_loss, self.epoch+1)
            self.writer.add_scalar('Val_Loss', val_loss, self.epoch+1)

            self.writer.add_scalar('Train_Perp', train_perp, self.epoch+1)
            self.writer.add_scalar('Val_Perp', val_perp, self.epoch+1)

            is_best = val_loss < self.best_val_loss
            self.best_val_loss = min(self.best_val_loss, val_loss)
            if self.options.save_model:
                self.save_model(is_best)
            self.logger.info(
                f'Epoch = [{1 + self.epoch:3d}/{self.options.num_epochs:3d}] Train_Loss = {train_loss:.4g} '
                f'Val_Loss = {val_loss:.4g} Train_Time = {train_time:.4f}s Val_Time = {val_time:.4f}s',
            )
            if is_best:
                self.best_epoch = epoch
            print('Best epoch: ',self.best_epoch)
            print('Waiting: ', epoch-self.best_epoch)
            if (epoch-self.best_epoch) >= self.options.early_stop:
                print("######### Early Stop #########")
                break
        self.writer.close()

    def train_epoch(self, epoch):
        self.model.train()
        if self.options.enc_freeze == True:
            self.model.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False


        for iter, data in enumerate(self.train_loader):
            start_iter = time.perf_counter()
            signal = data[0].to(self.options.device).float()
            label = data[1].to(self.options.device).long()
            
            batch_size = label.shape[0]
            seq_len = label.shape[1]  # Assuming label has shape (batch, seq_len)

            pos_input = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)
            pos_target = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)

            # Assign values for shifted positions
            pos_input[:, 0] = pos_tags["START"]
            pos_input[:, 1:seq_len+1] = label
            pos_target[:, :seq_len] = label

            # Find the first occurrence of PAD and replace with END
            mask = label == pos_tags["PAD"]
            indices = mask.int().argmax(dim=1)  # First occurrence of PAD along sequence dimension
            valid_indices = mask.any(dim=1)  # Check if PAD exists in each sequence
            pos_target[valid_indices, indices[valid_indices]] = pos_tags["END"]

            pos_input_one_hot = F.one_hot(pos_input, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)
            pos_target_one_hot = F.one_hot(pos_target, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)

            #print(pos_input[0,:10])
            #print(pos_target[0,:10])

            if self.options.model == 'TCAE':        
                output = self.model(signal, pos_input_one_hot)
                #print("Shapes:")
                #print(output.shape)
                #print(pos_target_one_hot.shape)

                loss = self.loss(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1), self.options.pred_loss_type)
                #total_loss.update(loss.item())
            else:
                print("Not Supported")

            #acc = accuracy(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1))
            #pred_acc.update(acc[0].item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #if iter % self.options.report_period == 0:
            #    print('Epoch[{0}][{1}/{2}]\t'
            #        'Time {batch_time.avg:.3f} ({batch_time.val:.3f})\t' 
            #        'Loss {total_loss.avg:.4f} ({total_loss.val:.4f})\t'
            #        'Acc {pred_acc.avg:.4f} ({pred_acc.val:.4f})'.format(
            #        epoch, iter, len(self.train_loader), batch_time = batch_time, total_loss = total_loss, pred_acc = pred_acc))

            #batch_time.update(time.perf_counter()-start_iter)
               
        #self.writer.add_scalar('Train_Loss', total_loss.avg, self.epoch)
        #print('Train Time {0} \t Train Loss {1}'.format(time.perf_counter() - start_epoch, total_loss.avg))

        #return total_loss.avg, time.perf_counter() - start_epoch
        return
    def evaluate(self, loader, calc_loss=True, calc_acc=True, calc_perp=True):
        self.model.eval()

        total_loss = AverageMeter()
        total_perp = AverageMeter()
        total_acc = AverageMeter()
        start_epoch = time.perf_counter()

        with torch.no_grad():
            for iter, data in enumerate(loader):
                signal = data[0].to(self.options.device).float()
                label = data[1].to(self.options.device).long()
                
                batch_size = label.shape[0]
                seq_len = label.shape[1]  # Assuming label has shape (batch, seq_len)

                pos_input = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)
                pos_target = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)

                # Assign values for shifted positions
                pos_input[:, 0] = pos_tags["START"]
                pos_input[:, 1:seq_len+1] = label
                pos_target[:, :seq_len] = label

                # Find the first occurrence of PAD and replace with END
                mask = label == pos_tags["PAD"]
                indices = mask.int().argmax(dim=1)  # First occurrence of PAD along sequence dimension
                valid_indices = mask.any(dim=1)  # Check if PAD exists in each sequence
                pos_target[valid_indices, indices[valid_indices]] = pos_tags["END"]

                pos_input_one_hot = F.one_hot(pos_input, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)
                pos_target_one_hot = F.one_hot(pos_target, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)


                if self.options.model == 'TCAE':        
                    output = self.model(signal, pos_input_one_hot)
                    #print("Shapes:")
                    #print(output.shape)
                    #print(pos_target_one_hot.shape)
                    if calc_loss:
                        loss = self.loss(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1), self.options.pred_loss_type)
                        total_loss.update(loss.item())
                    if calc_acc:
                        acc = multiclass_accuracy(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1), self.options.l_vocab, ignore_index=pos_tags["PAD"])
                        total_acc.update(acc.item())
                    if calc_perp:
                        perp = perplexity(output, pos_target, ignore_index=pos_tags["PAD"])
                        total_perp.update(perp.item())
                else:
                    print("Not Supported")

        out = {}
        if calc_loss:
            out['loss'] = total_loss.avg
        if calc_acc:
            out['acc'] = total_acc.avg
        if calc_perp:
            out['perp'] = total_perp.avg
        out['time'] = time.perf_counter() - start_epoch
        return out

    def loss(self, input, target, loss_type):
        """
        Args:
            input: reconstructed time series
            target: original time series
            loss_type: the type of loss function         
        """
        # Reconstruction Loss
        if loss_type == 'cross_entropy':
            # print('input: ',input.shape)
            # print('target: ',np.unique(target.detach().cpu().numpy()))
            loss = F.cross_entropy(input,target,ignore_index=pos_tags["PAD"])
        #elif loss_type == 'mse':
        #    loss = F.mse_loss(input,target)
        else:
            raise NotImplementedError()

        return loss

    def load(self):
        self.model = build_pos_predictor(self.options)
        if self.options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        checkpoint_tmp = torch.load(self.options.checkpoint)
        print("Load checkpoint {} with loss {}".format(checkpoint_tmp['epoch'], checkpoint_tmp['best_val_loss']))
        self.best_val_loss = checkpoint_tmp['best_val_loss']
        self.best_epoch = checkpoint_tmp['best_epoch']
        self.start_epoch = checkpoint_tmp['epoch'] + 1
        
        self.model.load_state_dict(checkpoint_tmp['model'])

        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.optimizer.load_state_dict(checkpoint_tmp['optimizer'])
        self.scheduler = build_lr_scheduler(self.options, self.optimizer,self.start_epoch)
        

    def save_model(self,is_best):
        exp_dir = self.options.exp_dir
        torch.save(
            {
                'epoch': self.epoch,
                'options': self.options,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'exp_dir': exp_dir,
                'best_epoch': self.best_epoch  # Save the best epoch
            },
            f = exp_dir / 'predictor_model.pt'
        )
        if is_best:
            shutil.copyfile(exp_dir / 'predictor_model.pt', exp_dir / 'best_predictor_model.pt')

class Predictor_Tester:
    def __init__(self, options, data_module, exp_dir):

        # setup the dataloader
        _, _, self.test_loader = data_module._setup_dataloaders()

        # load options and model
        embed = os.path.join(exp_dir ,'best_model.pt')
        pred = os.path.join(exp_dir ,'best_predictor_model.pt')
        print(exp_dir)
        self.load(embed,pred)
        self.options.exp_dir = exp_dir

    def __call__(self):
        self.evaluate()

    def evaluate(self):

        pred_acc = AverageMeter()
        pred_p = AverageMeter()
        pred_r = AverageMeter()
        pred_f1 = AverageMeter()
        pred_auc = AverageMeter()
        start_epoch = time.perf_counter()
        Emd = []
        Attention = []

        with torch.no_grad():
            for iter, data in enumerate(self.test_loader):
                start_iter = time.perf_counter()
                signal = data[0].to(self.options.device).float()
                label = data[1].to(self.options.device).long()

                if self.options.model == 'DVAE' or self.options.model == 'DRVAE':
                    embed, mu, log_var = self.embed.encode(signal)
                    output = self.pred(embed)
                elif self.options.model == 'TCAE':
                    embed = self.embed.encode(signal,None)
                    attn = self.embed.encoder.layers[0].self_attn.attn.data.cpu()
                    output = self.pred(embed)
                else:
                    embed = self.embed.encode(signal)
                    output = self.pred(embed)

                accuracy = Accuracy().to(self.options.device)
                precision = Precision(num_classes=self.options.num_class, average='macro').to(self.options.device)
                recall = Recall(num_classes=self.options.num_class, average='macro').to(self.options.device)
                f1score= F1Score(num_classes=self.options.num_class, average='macro').to(self.options.device)
                aucroc = AUROC(num_classes=self.options.num_class).to(self.options.device)

                acc = accuracy(output.reshape(-1,self.options.num_class), label.reshape(-1))
                p = precision(output.reshape(-1,self.options.num_class), label.reshape(-1))
                r = recall(output.reshape(-1,self.options.num_class), label.reshape(-1))
                f1 = f1score(output.reshape(-1,self.options.num_class), label.reshape(-1))
                auc = aucroc(output.reshape(-1,self.options.num_class), label.reshape(-1))

                pred_acc.update(acc.item())
                pred_p.update(p.item())
                pred_r.update(r.item())
                pred_f1.update(f1.item())
                pred_auc.update(auc.item())

                Emd.append(embed.detach().cpu().numpy())
                # Attention.append(attn)

            if self.options.save_embedding:
                Emd = np.concatenate(Emd,axis=0)
                # Attention = np.concatenate(Attention,axis = 0)
                exp_dir = self.options.exp_dir
                np.save(os.path.join(exp_dir ,'test_embeds.npy'), Emd)
                # np.save(os.path.join(exp_dir ,'attn_map.npy'), Attention)
                print('Test embeddings saved')

            print('Test Time {0:.4f} \nTest Acc {1:.4f} \nTest F1 {2:.4f} \nTest P {3:.4f} \nTest R {4:.4f} \nTest AUC {5:.4f}'.\
                format(time.perf_counter() - start_epoch, pred_acc.avg, pred_f1.avg, pred_p.avg, pred_r.avg, pred_auc.avg))
            # print('Test Acc {0:.6f} \nTest F1 {1:.6f} \nTest P {2:.6f} \nTest R {3:.6f} \nTest AUC {4:.6f}'.\
            #     format(pred_acc.avg, pred_f1.avg, pred_p.avg, pred_r.avg, pred_auc.avg))

    def load(self, embed_checkpoint_file, pred_checkpoint_file):

        # embedding model
        embed_checkpoint = torch.load(embed_checkpoint_file)
        print("Load embedding checkpoint {} with loss {}".format(embed_checkpoint['epoch'], embed_checkpoint['best_val_loss']))
        self.options = embed_checkpoint['options']
        self.embed = build_model(self.options)
        if self.options.data_parallel:
            self.embed = torch.nn.DataParallel(self.embed)
            self.embed.device_ids = [0,1]
        self.embed.eval()
        self.embed.load_state_dict(embed_checkpoint['model'])

        # prediction model
        pred_checkpoint = torch.load(pred_checkpoint_file)
        print("Load prediction checkpoint {} with loss {}".format(pred_checkpoint['epoch'], pred_checkpoint['best_val_loss']))
        self.options = pred_checkpoint['options']
        self.pred = build_predictor(self.options)
        if self.options.data_parallel:
            self.pred = torch.nn.DataParallel(self.pred)
            self.pred.device_ids = [0,1]
        self.pred.eval()
        self.pred.load_state_dict(pred_checkpoint['model'])

    def save(self, fname, array):
        np.save(fname, array)


class LM_Pretrainer:
    """
    Trainer.

    Configuration for the POS decoder trainer is provided by the argument 'options'. 
    Must contain the following fields:

    Args:
        options('argparse.Namespace'): Options for the trainer.
        data_module('object'): Data module class for the used datasets.
    """
    def __init__(self, options, data_module):
        self.options = options

        # setup model, optimizer and scheduler
        self.model = build_pos_decoder(self.options)
        self.model = self.model.to(self.options.device)

        if options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.model.device_ids = [0,1]
        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.scheduler = build_lr_scheduler(self.options, self.optimizer)
        

        # setup the dataloader
        self.train_loader, self.val_loader, _ = data_module._setup_dataloaders()

        self.epoch = 0
        self.start_epoch = 0
        self.end_epoch = options.num_epochs

        self.best_val_loss = np.inf
        self.best_epoch = 0

        # setup saving, writer, and logging
        options.exp_dir.mkdir(parents=True, exist_ok=True)
        options.inference_dir.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(str(options.exp_dir), 'args.pkl'), "wb") as f:
            pickle.dump(options.__dict__, f)

        self.writer = SummaryWriter(log_dir=options.exp_dir / 'summary')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)
        self.logger.info(self.model)

    def load_checkpoint(self):
        if self.options.resume:
            self.load()

    def __call__(self):
        self.load_checkpoint()
        initial_val_loss, _ = self.evaluate_epoch()
        self.writer.add_scalar('Val_Loss', initial_val_loss, 0)  # Log initial validation loss at epoch 0
        return self.train()

    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            train_loss, train_time = self.train_epoch(self.epoch)
            self.scheduler.step()
            val_loss, val_time = self.evaluate_epoch()

            is_best = val_loss < self.best_val_loss
            self.best_val_loss = min(self.best_val_loss, val_loss)
            if self.options.save_model:
                self.save_model(is_best)
            self.logger.info(
                f'Epoch = [{1 + self.epoch:3d}/{self.options.num_epochs:3d}] Train_Loss = {train_loss:.4g} '
                f'Val_Loss = {val_loss:.4g} Train_Time = {train_time:.4f}s Val_Time = {val_time:.4f}s',
            )
            if is_best:
                self.best_epoch = epoch
            print('Best epoch: ',self.best_epoch)
            print('Waiting: ', epoch-self.best_epoch)
            if (epoch-self.best_epoch) >= self.options.early_stop:
                print("######### Early Stop #########")
                break
        self.writer.close()

    def train_epoch(self, epoch):
        self.model.train()
        if self.options.freeze == True:
            self.model.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        total_loss = AverageMeter()
        batch_time = AverageMeter()
        pred_acc = AverageMeter()
        start_epoch = time.perf_counter()

        for iter, data in enumerate(self.train_loader):
            start_iter = time.perf_counter()
            label = data.to(self.options.device).long()
            
            batch_size = label.shape[0]
            seq_len = label.shape[1]  # Assuming label has shape (batch, seq_len)

            pos_input = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)
            pos_target = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)

            # Assign values for shifted positions
            pos_input[:, 0] = pos_tags["START"]
            pos_input[:, 1:seq_len+1] = label
            pos_target[:, :seq_len] = label

            # Find the first occurrence of PAD and replace with END
            mask = label == pos_tags["PAD"]
            indices = mask.int().argmax(dim=1)  # First occurrence of PAD along sequence dimension
            valid_indices = mask.any(dim=1)  # Check if PAD exists in each sequence
            pos_target[valid_indices, indices[valid_indices]] = pos_tags["END"]

            pos_input_one_hot = F.one_hot(pos_input, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)
            pos_target_one_hot = F.one_hot(pos_target, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)

            #print(pos_input[0,:10])
            #print(pos_target[0,:10])

            if self.options.model == 'TCAE':
                batch_size = label.shape[0] 
                dummy_signal = torch.zeros(batch_size, 10, self.options.d_latent).to(self.options.device)   
                output = self.model(pos_input_one_hot, dummy_signal)
                #print("Shapes:")
                #print(output.shape)
                #print(pos_target_one_hot.shape)

                loss = self.loss(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1), self.options.pred_loss_type)
                total_loss.update(loss.item())
            else:
                print("Not Supported")

            acc = accuracy(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1))
            pred_acc.update(acc[0].item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if iter % self.options.report_period == 0:
                print('Epoch[{0}][{1}/{2}]\t'
                    'Time {batch_time.avg:.3f} ({batch_time.val:.3f})\t' 
                    'Loss {total_loss.avg:.4f} ({total_loss.val:.4f})\t'
                    'Acc {pred_acc.avg:.4f} ({pred_acc.val:.4f})'.format(
                    epoch, iter, len(self.train_loader), batch_time = batch_time, total_loss = total_loss, pred_acc = pred_acc))

            batch_time.update(time.perf_counter()-start_iter)
               
        self.writer.add_scalar('Train_Loss', total_loss.avg, self.epoch + 1)
        print('Train Time {0} \t Train Loss {1}'.format(time.perf_counter() - start_epoch, total_loss.avg))

        return total_loss.avg, time.perf_counter() - start_epoch

    def evaluate_epoch(self):
        self.model.eval()

        total_loss = AverageMeter()
        pred_acc = AverageMeter()
        start_epoch = time.perf_counter()

        with torch.no_grad():
            for iter, data in enumerate(self.val_loader):
                label = data.to(self.options.device).long()
                
                batch_size = label.shape[0]
                seq_len = label.shape[1]  # Assuming label has shape (batch, seq_len)

                pos_input = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)
                pos_target = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)

                # Assign values for shifted positions
                pos_input[:, 0] = pos_tags["START"]
                pos_input[:, 1:seq_len+1] = label
                pos_target[:, :seq_len] = label

                # Find the first occurrence of PAD and replace with END
                mask = label == pos_tags["PAD"]
                indices = mask.int().argmax(dim=1)  # First occurrence of PAD along sequence dimension
                valid_indices = mask.any(dim=1)  # Check if PAD exists in each sequence
                pos_target[valid_indices, indices[valid_indices]] = pos_tags["END"]

                pos_input_one_hot = F.one_hot(pos_input, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)
                pos_target_one_hot = F.one_hot(pos_target, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)


                if self.options.model == 'TCAE':       
                    batch_size = label.shape[0] 
                    dummy_signal = torch.zeros(batch_size, 10, self.options.d_latent).to(self.options.device)    
                    output = self.model(pos_input_one_hot, dummy_signal)
                    #print("Shapes:")
                    #print(output.shape)
                    #print(pos_target_one_hot.shape)

                    loss = self.loss(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1), self.options.pred_loss_type)
                    total_loss.update(loss.item())
                else:
                    print("Not Supported")

            acc = accuracy(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1))
            pred_acc.update(acc[0].item())

            self.writer.add_scalar('Val_Loss', total_loss.avg, self.epoch + 1)
            print('Val Time {0} \t Val Loss {1} \t Val Acc {2}'.format(time.perf_counter() - start_epoch, total_loss.avg, pred_acc.avg))

        return total_loss.avg, time.perf_counter() - start_epoch

    def loss(self, input, target, loss_type):
        """
        Args:
            input: reconstructed time series
            target: original time series
            loss_type: the type of loss function         
        """
        # Reconstruction Loss
        if loss_type == 'cross_entropy':
            # print('input: ',input.shape)
            # print('target: ',np.unique(target.detach().cpu().numpy()))
            loss = F.cross_entropy(input,target,ignore_index=pos_tags["PAD"])
        #elif loss_type == 'mse':
        #    loss = F.mse_loss(input,target)
        else:
            raise NotImplementedError()

        return loss

    def load(self):
        self.model = build_pos_decoder(self.options)
        if self.options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        checkpoint_tmp = torch.load(self.options.checkpoint)
        print("Load checkpoint {} with loss {}".format(checkpoint_tmp['epoch'], checkpoint_tmp['best_val_loss']))
        self.best_val_loss = checkpoint_tmp['best_val_loss']
        self.best_epoch = checkpoint_tmp['best_epoch']
        self.start_epoch = checkpoint_tmp['epoch'] + 1
        
        self.model.load_state_dict(checkpoint_tmp['model'])

        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.optimizer.load_state_dict(checkpoint_tmp['optimizer'])
        self.scheduler = build_lr_scheduler(self.options, self.optimizer,self.start_epoch)
        

    def save_model(self,is_best):
        exp_dir = self.options.exp_dir
        torch.save(
            {
                'epoch': self.epoch,
                'options': self.options,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'exp_dir': exp_dir,
                'best_epoch': self.best_epoch  # Save the best epoch
            },
            f = exp_dir / 'lm_pretrained_model.pt'
        )
        if is_best:
            shutil.copyfile(exp_dir / 'lm_pretrained_model.pt', exp_dir / 'best_lm_pretrained_model.pt')


class LM_Pretrainer_New:
    """
    Trainer.

    Configuration for the POS predictor trainer is provided by the argument 'options'. 
    Must contain the following fields:

    Args:
        options('argparse.Namespace'): Options for the trainer.
        data_module('object'): Data module class for the used datasets.
    """
    def __init__(self, options, data_module):
        self.options = options

        # setup model, optimizer and scheduler
        self.model = build_pos_predictor(self.options)
        self.model = self.model.to(self.options.device)

        if options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.model.device_ids = [0,1]
        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.scheduler = build_lr_scheduler(self.options, self.optimizer)
        

        # setup the dataloader
        self.train_loader, self.val_loader, _ = data_module._setup_dataloaders()

        self.epoch = 0
        self.start_epoch = 0
        self.end_epoch = options.num_epochs

        self.best_val_loss = np.inf
        self.best_epoch = 0

        # setup saving, writer, and logging
        options.exp_dir.mkdir(parents=True, exist_ok=True)
        options.inference_dir.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(str(options.exp_dir), 'args.pkl'), "wb") as f:
            pickle.dump(options.__dict__, f)

        self.writer = SummaryWriter(log_dir=options.exp_dir / 'summary')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)
        self.logger.info(self.model)

    def load_checkpoint(self):
        if self.options.resume:
            self.load()

    def __call__(self):
        self.load_checkpoint()
        return self.train()

    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train_epoch(self.epoch)
            self.scheduler.step()
            eval_res = self.evaluate(self.val_loader)
            val_loss = eval_res["loss"]

            is_best = val_loss < self.best_val_loss
            self.best_val_loss = min(self.best_val_loss, val_loss)
            if self.options.save_model:
                self.save_model(is_best)
            self.logger.info(
                f'Epoch = [{1 + self.epoch:3d}/{self.options.num_epochs:3d}] Train_Loss = {train_loss:.4g} '
                f'Val_Loss = {val_loss:.4g} Train_Time = {train_time:.4f}s Val_Time = {val_time:.4f}s',
            )
            if is_best:
                self.best_epoch = epoch
            print('Best epoch: ',self.best_epoch)
            print('Waiting: ', epoch-self.best_epoch)
            if (epoch-self.best_epoch) >= self.options.early_stop:
                print("######### Early Stop #########")
                break
        self.writer.close()

    def train_epoch(self, epoch):
        self.model.train()
        if self.options.freeze == True:
            self.model.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False


        for iter, data in enumerate(self.train_loader):
            print(f"Batch: {iter}")
            start_iter = time.perf_counter()
            label = data.to(self.options.device).long()
            
            batch_size = label.shape[0]
            seq_len = label.shape[1]  # Assuming label has shape (batch, seq_len)

            pos_input = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)
            pos_target = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)

            # Assign values for shifted positions
            pos_input[:, 0] = pos_tags["START"]
            pos_input[:, 1:seq_len+1] = label
            pos_target[:, :seq_len] = label

            # Find the first occurrence of PAD and replace with END
            mask = label == pos_tags["PAD"]
            indices = mask.int().argmax(dim=1)  # First occurrence of PAD along sequence dimension
            valid_indices = mask.any(dim=1)  # Check if PAD exists in each sequence
            pos_target[valid_indices, indices[valid_indices]] = pos_tags["END"]

            pos_input_one_hot = F.one_hot(pos_input, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)
            pos_target_one_hot = F.one_hot(pos_target, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)

            #print(pos_input[0,:10])
            #print(pos_target[0,:10])

            if self.options.model == 'TCAE':        
                batch_size = label.shape[0] 
                dummy_signal = torch.zeros(batch_size, 10, self.options.d_latent).to(self.options.device)   
                output = self.model(pos_input_one_hot, dummy_signal)
                #print("Shapes:")
                #print(output.shape)
                #print(pos_target_one_hot.shape)

                loss = self.loss(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1), self.options.pred_loss_type)
            else:
                print("Not Supported")

            #acc = accuracy(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1))
            #pred_acc.update(acc[0].item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if iter % self.options.report_period == 0:
                eval_res = self.evaluate(self.val_loader)
                val_loss = eval_res["loss"]
                val_perp = eval_res["perp"]
                val_time = eval_res ["time"]

                eval_tres = self.evaluate(self.train_loader)
                train_loss = eval_tres["loss"]
                train_perp = eval_tres["perp"]
                train_time = eval_tres ["time"]

                self.writer.add_scalar('Train_Loss', train_loss, iter)
                self.writer.add_scalar('Val_Loss', val_loss, iter)

                self.writer.add_scalar('Train_Perp', train_perp, iter)
                self.writer.add_scalar('Val_Perp', val_perp, iter)

                print(f"Logging for batch {iter}")

            #batch_time.update(time.perf_counter()-start_iter)
               
        #self.writer.add_scalar('Train_Loss', total_loss.avg, self.epoch)
        #print('Train Time {0} \t Train Loss {1}'.format(time.perf_counter() - start_epoch, total_loss.avg))

        #return total_loss.avg, time.perf_counter() - start_epoch
        return
    def evaluate(self, loader, calc_loss=True, calc_acc=True, calc_perp=True):
        self.model.eval()

        total_loss = AverageMeter()
        total_perp = AverageMeter()
        total_acc = AverageMeter()
        start_epoch = time.perf_counter()

        with torch.no_grad():
            for iter, data in enumerate(loader):
                label = data.to(self.options.device).long()
            
                batch_size = label.shape[0]
                seq_len = label.shape[1]  # Assuming label has shape (batch, seq_len)

                pos_input = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)
                pos_target = torch.full((batch_size, seq_len + 1), pos_tags["PAD"], dtype=torch.long, device=self.options.device)

                # Assign values for shifted positions
                pos_input[:, 0] = pos_tags["START"]
                pos_input[:, 1:seq_len+1] = label
                pos_target[:, :seq_len] = label

                # Find the first occurrence of PAD and replace with END
                mask = label == pos_tags["PAD"]
                indices = mask.int().argmax(dim=1)  # First occurrence of PAD along sequence dimension
                valid_indices = mask.any(dim=1)  # Check if PAD exists in each sequence
                pos_target[valid_indices, indices[valid_indices]] = pos_tags["END"]

                pos_input_one_hot = F.one_hot(pos_input, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)
                pos_target_one_hot = F.one_hot(pos_target, num_classes=20).float()   # Shape: (batch_size, seq_len+1, 20)


                if self.options.model == 'TCAE':        
                    batch_size = label.shape[0] 
                    dummy_signal = torch.zeros(batch_size, 10, self.options.d_latent).to(self.options.device)   
                    output = self.model(pos_input_one_hot, dummy_signal)
                    #print("Shapes:")
                    #print(output.shape)
                    #print(pos_target_one_hot.shape)
                    if calc_loss:
                        loss = self.loss(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1), self.options.pred_loss_type)
                        total_loss.update(loss.item())
                    if calc_acc:
                        acc = multiclass_accuracy(output.reshape(-1,len(pos_tags)), pos_target.reshape(-1), self.options.l_vocab, ignore_index=pos_tags["PAD"])
                        total_acc.update(acc.item())
                    if calc_perp:
                        perp = perplexity(output, pos_target, ignore_index=pos_tags["PAD"])
                        total_perp.update(perp.item())
                else:
                    print("Not Supported")

        out = {}
        if calc_loss:
            out['loss'] = total_loss.avg
        if calc_acc:
            out['acc'] = total_acc.avg
        if calc_perp:
            out['perp'] = total_perp.avg
        out['time'] = time.perf_counter() - start_epoch
        return out

    def loss(self, input, target, loss_type):
        """
        Args:
            input: reconstructed time series
            target: original time series
            loss_type: the type of loss function         
        """
        # Reconstruction Loss
        if loss_type == 'cross_entropy':
            # print('input: ',input.shape)
            # print('target: ',np.unique(target.detach().cpu().numpy()))
            loss = F.cross_entropy(input,target,ignore_index=pos_tags["PAD"])
        #elif loss_type == 'mse':
        #    loss = F.mse_loss(input,target)
        else:
            raise NotImplementedError()

        return loss

    def load(self):
        self.model = build_pos_predictor(self.options)
        if self.options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        checkpoint_tmp = torch.load(self.options.checkpoint)
        print("Load checkpoint {} with loss {}".format(checkpoint_tmp['epoch'], checkpoint_tmp['best_val_loss']))
        self.best_val_loss = checkpoint_tmp['best_val_loss']
        self.best_epoch = checkpoint_tmp['best_epoch']
        self.start_epoch = checkpoint_tmp['epoch'] + 1
        
        self.model.load_state_dict(checkpoint_tmp['model'])

        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.optimizer.load_state_dict(checkpoint_tmp['optimizer'])
        self.scheduler = build_lr_scheduler(self.options, self.optimizer,self.start_epoch)
        

    def save_model(self,is_best):
        exp_dir = self.options.exp_dir
        torch.save(
            {
                'epoch': self.epoch,
                'options': self.options,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'exp_dir': exp_dir,
                'best_epoch': self.best_epoch  # Save the best epoch
            },
            f = exp_dir / 'lm_pre_model.pt'
        )
        if is_best:
            shutil.copyfile(exp_dir / 'lm_pre_model.pt', exp_dir / 'best_lm_pre_model.pt')
