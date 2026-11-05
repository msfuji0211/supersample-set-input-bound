import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloaders.utils.image_utils import ImageDatasetWrapper
from configs.datasets import dataset_info
from transformers import ViTForImageClassification
from torchmetrics import Accuracy, CalibrationError
import shutil
import numpy as np
import os


class ViT(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        outputs = super().__call__(pixel_values=x)[0]
        return outputs


class ViTClassifier():
    def __init__(self, SAVE_DIR, config, from_saved=False, saved_epoch=0, save_scheme='best-val-acc'):
        self.SAVE_DIR = SAVE_DIR
        self.from_saved = from_saved
        self.saved_epoch = saved_epoch
        self.save_scheme = save_scheme

        # choose device, initialize variables
        self.load_config(config)

        # load model and training objects
        self.net = self.load_net(from_saved, saved_epoch)
        print("ViT loaded.")

    def load_net(self, from_saved, epoch=0):
        if from_saved:
            sd = self.SAVE_DIR + f'model_{epoch}'
            net = ViT.from_pretrained(sd, num_labels=2)
            print(f'Using model saved at {sd}.')
        else:
            net = ViT.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=2)
        return net.to(self.device)


    def load_config(self, config):
        # eval parameters
        self.config = config
        self.batch_size = config['batch_size']
        self.train_log_interval = int(config.get('train_log_interval', 5000))
        self.loss_log_interval = int(config.get('loss_log_interval', 100))

        # pretrained ViT expects 224x224 images, so change target_res to (224, 224)
        self.transform_config = dataset_info(self.config['dataset'])
        self.transform_config['target_res'] = (224, 224)

        # training
        self.NUM_WORKERS = int(config.get('num_workers', 0))
        self.PIN_MEMORY = bool(config.get('pin_memory', True))
        
        # determine device
        self.device = "cpu"
        # check cuda
        if torch.cuda.is_available():
            self.device = "cuda"
            print("Setting device = cuda.")
        # check for MPS (Mac M1)
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("Setting device = MPS.")
        elif not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
        # cpu default
        else:
            print("No device found; setting device = cpu.")


    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        # training parameters
        (
            self.optim,
            self.epochs,
            self.lr_schedule,
            self.weight_decay,
            self.val_save_epoch,
            self.val_eval_epoch,
        ) = (
            self.config['optim'],
            self.config['epochs'],
            self.config['lr_schedule'],
            self.config['weight_decay'],
            self.config['val_save_epoch'],
            self.config['val_eval_epoch'],
        )

        # if SAVE_DIR is not a directory yet, create it
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        # create dataloaders
        train_dataset = ImageDatasetWrapper(X_train, y_train, self.transform_config)
        val_dataset = ImageDatasetWrapper(X_val, y_val, self.transform_config)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=self.NUM_WORKERS,
                                                       pin_memory=self.PIN_MEMORY,
                                                       drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=False,
                                                     num_workers=self.NUM_WORKERS,
                                                     pin_memory=self.PIN_MEMORY,
                                                     drop_last=False)
        print(
            f"[train] n_train={len(train_dataset)} n_val={len(val_dataset)} "
            f"batch_size={self.batch_size} batches_per_epoch={len(train_dataloader)} "
            f"epochs={self.config['epochs']} num_workers={self.NUM_WORKERS} "
            f"loss_log_interval={self.loss_log_interval} metric_log_interval={self.train_log_interval}"
        )

        # define loss function and optimiser
        criterion = nn.CrossEntropyLoss().to(self.device)
        if self.optim == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), 
                                  lr=self.lr_schedule[0], 
                                  weight_decay=self.weight_decay)
        elif self.optim == 'adam':
            optimizer = optim.Adam(self.net.parameters(), 
                                   lr=self.lr_schedule[0], 
                                   weight_decay=self.weight_decay)
        else:
            raise ValueError('Optimizer not supported.')
        
        train_loss = []
        best_val_acc = -1
        accuracy = Accuracy(task='multiclass', num_classes=2).to(self.device)
        ece = CalibrationError(task='multiclass', num_classes=2, n_bins=10).to(self.device)

        # determine epochs
        if self.from_saved:
            # if saved epoch is already the last epoch, raise error
            if self.saved_epoch >= self.epochs - 1:
                raise ValueError('Model already trained for', self.saved_epoch, 'epochs.')
            
            epoch_range = range(self.saved_epoch + 1, self.epochs)

            # find current learning rate
            for lr_epoch in self.lr_schedule:
                if lr_epoch <= self.saved_epoch:
                    current_lr = self.lr_schedule[lr_epoch]
                else: break

        else: epoch_range = range(self.epochs)

        # train loop
        self.recent_save = None
        for epoch in epoch_range:
            self.net.train()
            running_loss = 0.0
            running_acc = 0.0
            running_ece = 0.0
            num_batches = 0
            metric_batches = 0

            # update learning rate
            if epoch in self.lr_schedule:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr_schedule[epoch]
                    current_lr = param_group['lr']

            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # backprop
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                num_batches += 1
                running_loss += loss.item()

                should_log_loss = (
                    self.loss_log_interval > 0
                    and ((i + 1) % self.loss_log_interval == 0 or (i + 1) == len(train_dataloader))
                )
                if should_log_loss:
                    print('[%d, %5d/%5d] loss: %.3f LR: %.5f' %
                        (epoch + 1, i + 1, len(train_dataloader), running_loss / num_batches, current_lr))

                # Accuracy/ECE are diagnostics only. Computing CalibrationError
                # every mini-batch is relatively expensive for repeated
                # supersample retraining, so compute it only at the requested
                # logging interval (default: 5000).
                should_log_metrics = (
                    self.train_log_interval > 0
                    and ((i + 1) % self.train_log_interval == 0 or (i + 1) == len(train_dataloader))
                )
                if should_log_metrics:
                    out_d = F.softmax(outputs.detach(), dim=1)
                    labels_d = labels.detach()
                    curr_acc = accuracy(out_d, labels_d)
                    curr_ece = ece(out_d, labels_d)
                    running_acc += float(curr_acc.detach().cpu())
                    running_ece += float(curr_ece.detach().cpu())
                    metric_batches += 1

                    train_loss.append(running_loss / num_batches)
                    print('[%d, %5d/%5d] metrics: sampled acc %.5f sampled ECE %.5f' %
                        (epoch + 1, i + 1, len(train_dataloader),
                         running_acc / metric_batches, running_ece / metric_batches))

            if metric_batches > 0:
                print('Epoch:', epoch + 1, 'Loss:', running_loss,
                      'Sampled acc:', running_acc / metric_batches,
                      'Sampled ECE:', running_ece / metric_batches)
            else:
                print('Epoch:', epoch + 1, 'Loss:', running_loss,
                      'Sampled acc/ECE: not computed')

            # Validate/checkpoint only when explicitly requested.  In Eq.33 supersample
            # runs the default is save_scheme='final', so the complementary branch is not
            # used to choose W.
            if self.save_scheme != 'final' and epoch % self.val_eval_epoch == 0:
                self.net.eval()

                with torch.no_grad():
                    ## validation split
                    # get predictions
                    (y_conf, y_logits), y_val = self.evaluate_loader(val_dataloader)
                    y_pred = torch.argmax(y_conf, axis=1)

                    # metrics
                    print('val: y-pred mean', round(torch.mean(y_pred.type(torch.float32)).item(), 3))
                    print('val: y-true mean', round(torch.mean(y_val.type(torch.float32)).item(), 3))
                    val_acc = accuracy(y_val, y_pred).item()
                    print('val: acc', round(val_acc, 3))
                    
                    # save every epoch, according to save_scheme
                    self.save_model(epoch, val_acc, best_val_acc)
                    best_val_acc = max(val_acc, best_val_acc)

        # save config to file
        with open(self.SAVE_DIR + 'config.txt', 'w') as f:
            f.write(str(self.config))

        if self.save_scheme == 'final':
            final_epoch = list(epoch_range)[-1]
            self.recent_save = final_epoch
            self.net.save_pretrained(self.SAVE_DIR + f'model_{final_epoch}')
            print(f'Saved final-epoch model: {self.SAVE_DIR}model_{final_epoch}')
        else:
            print('Best validation accuracy:', best_val_acc)

        # load selected model
        self.net = ViT.from_pretrained(self.SAVE_DIR + f'model_{self.recent_save}', num_labels=2).to(self.device)

    def save_model(self, epoch, val_acc, old_best_val_acc):
        '''save model according to save_scheme'''
        if epoch < self.val_save_epoch: return
        elif self.recent_save == None:
            self.recent_save = epoch
            self.net.save_pretrained(self.SAVE_DIR + f'model_{epoch}')
            return
        
        # all-epochs: save every epoch
        if self.save_scheme == 'all-epochs':
            self.recent_save = epoch    
            self.net.save_pretrained(self.SAVE_DIR + f'model_{epoch}')

        # best-val-acc: save only if val_acc is better than old_best_val_acc
        elif self.save_scheme == 'best-val-acc':
            self.recent_save = epoch
            if val_acc > old_best_val_acc:
                self.net.save_pretrained(self.SAVE_DIR + f'model_{epoch}')
            else:
                src = self.SAVE_DIR + f'model_{epoch - self.val_eval_epoch}'
                dst = self.SAVE_DIR + f'model_{epoch}'
                # ViT checkpoints saved by HuggingFace are directories.
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
    
    def evaluate_loader(self, dataloader):
        '''
        Evaluate model on dataloader, return detatched confidence/labels.
        '''
        all_confs, all_logits, all_lbls = [], [], []

        for i, batch in enumerate(dataloader):
            # inputs, labels = batch['pixel_values'], batch['labels']
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs).detach()
            p = F.softmax(outputs, dim=1)

            all_logits.append(outputs)
            all_confs.append(p)
            all_lbls.append(labels)

        return (torch.cat(all_confs), torch.cat(all_logits)), torch.cat(all_lbls)
    

    def predict_proba(self, X_idxs, with_logits=False):
        '''
        Prediction function for evaluation.
        Used only by Experiment.py class.
        '''
        ds = ImageDatasetWrapper(X_idxs, y=None, config=self.transform_config)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, 
                                         shuffle=False,
                                         num_workers=self.NUM_WORKERS,
                                         pin_memory=self.PIN_MEMORY,
                                         drop_last=False)
        
        (confs, logits), labels = self.evaluate_loader(dl)
        if with_logits: return confs.cpu().numpy(), logits.cpu().numpy()
        else: return confs.cpu().numpy()


    def predict(self, X_text):
        '''
        Prediction function for evaluation.
        Used only by Experiment.py class.
        '''
        confs = self.predict_proba(X_text)
        preds = np.argmax(confs, axis=1)

        return preds