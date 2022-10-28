from tqdm import tqdm
import torch
import numpy as np
import torch.distributed as dist
from torch.cuda.amp import autocast,GradScaler
from utils.ddp_util import all_gather
import utils.metric as metric

class ModelTrainer:
    args = None
    model = None
    classification_head = None
    logger = None
    criterion = None
    optimizer = None
    lr_scheduler = None
    train_set = None
    train_set_sampler = None
    train_loader = None
    validation_set = None
    validation_loader = None
    evaluation_set = None
    evaluation_loader = None
    spec = None

    def run(self):
        
        self.best_acc = {}
        self.best_acc['val_fold_{}'.format(self.fold)] = -np.inf
        self.best_acc['eval_fold_{}'.format(self.fold)] = -np.inf

        self.cum_predictions = {}
        
        self.scaler = GradScaler()
        for epoch in range(1, self.args.epoch+1):
            self.train_set_sampler.set_epoch(epoch)
            self.train(epoch)
            self._synchronize()
            self.test(epoch, 'eval', self.evaluation_loader)
            self._synchronize()
        
    def train(self, epoch):
        self.model.train()
        self.classification_head.train()
        idx_ct_start = len(self.train_loader)*(int(epoch)-1)

        _loss = 0.

        with tqdm(total = len(self.train_loader), ncols = 150) as pbar:
            for idx, (m_batch,  m_label) in enumerate(self.train_loader):
                
                global_step = idx + idx_ct_start                
                m_batch = m_batch.to(self.args.device, non_blocking=True)
                m_label = m_label.to(self.args.device, non_blocking=True)
                
        
                description = '%s fold: %d epoch: %d '%(self.args.name, self.fold, epoch)

                with autocast():
                    embedding = self.model(m_batch)
                    output = self.classification_head(embedding)
                    if isinstance(self.criterion['classification_loss'], torch.nn.CrossEntropyLoss):
                        loss = self.criterion['classification_loss'](output, torch.argmax(m_label.long(), axis=1))
                    else:
                        loss = self.criterion['classification_loss'](output, m_label)
                        
                _loss += loss.cpu().detach() 
                description += 'loss:%.3f '%(loss)
                
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()


                pbar.set_description(description)
                pbar.update(1)

                # if the current epoch is match to the logging condition, log
                if idx % self.args.number_iteration_for_log == 0:
                    if idx != 0:
                        _loss /= self.args.number_iteration_for_log
                    
                        for p_group in self.optimizer.param_groups:
                            lr = p_group['lr']
                            break

                        if self.args.flag_parent:
                            self.logger.log_metric('loss_fold_{}'.format(self.fold), _loss, step = idx_ct_start+idx)
                            self.logger.log_metric('lr_fold_{}'.format(self.fold), lr, step = idx_ct_start+idx)
                            _loss = 0.
                            
        self.lr_scheduler.step()        

    def test(self, epoch, mode, dataloader):
        with self.model.no_sync():
            predictions, labels, eval_loss = self._evaluate(dataloader) 

        if epoch == 1: self.cum_predictions[mode] = predictions
        else: self.cum_predictions[mode] = self.cum_predictions[mode] * (epoch - 1) + predictions

        self.cum_predictions[mode] = self.cum_predictions[mode] / epoch

        if self.args.flag_parent:
            stats = metric.calculate_statistics(predictions, labels)
            cum_stats = metric.calculate_statistics(self.cum_predictions[mode], labels)

            acc, mAP, mAUC, d_prime, average_precision, average_recall, cum_mAP, cum_mAUC, cum_acc = metric.calculate_metric(stats, cum_stats)

            self.logger.log_metric('{}_acc_{}'.format(mode, self.fold), acc, epoch_step=epoch)
            self.logger.log_metric('{}_mAP_{}'.format(mode, self.fold), mAP, epoch_step=epoch)
            self.logger.log_metric('{}_mAUC_{}'.format(mode, self.fold), mAUC, epoch_step=epoch)
            self.logger.log_metric('{}_d_prime_{}'.format(mode, self.fold), d_prime, epoch_step=epoch)
            self.logger.log_metric('{}_average_precision_{}'.format(mode, self.fold), average_precision, epoch_step=epoch)
            self.logger.log_metric('{}_average_recall_{}'.format(mode, self.fold), average_recall, epoch_step=epoch)
            self.logger.log_metric('{}_cum_mAP_{}'.format(mode, self.fold), cum_mAP, epoch_step=epoch)
            self.logger.log_metric('{}_cum_mAUC_{}'.format(mode, self.fold), cum_mAUC, epoch_step=epoch)
            self.logger.log_metric('{}_cum_acc_{}'.format(mode, self.fold), cum_acc, epoch_step=epoch)
            self.logger.log_metric('{}_eval_loss_{}'.format(mode, self.fold), eval_loss, epoch_step=epoch)
            
            self.results[self.fold-1, epoch-1, :] = [acc, mAP, mAUC, d_prime, average_precision, average_recall, cum_mAP, cum_mAUC, cum_acc, eval_loss]

            if acc > self.best_acc[mode+'_fold_{}'.format(self.fold)]:
                self.best_acc[mode+'_fold_{}'.format(self.fold)] = acc
                self.logger.log_metric('Bestacc_{}_{}'.format(mode, self.fold), acc, epoch_step=epoch)

    def _evaluate(self, dataloader):
        self.model.eval()
        self.classification_head.eval()
        
        l_predictions = []
        l_labels = []
        l_losses = []
        l_paths = []

        with torch.set_grad_enabled(False):
            for m_batch, labels, path in tqdm(dataloader, ncols=120):
                m_batch = m_batch.to(self.args.device, non_blocking=True)
                bs = m_batch.shape[0]
                m_batch = m_batch.view(-1, self.args.frame_length, self.args.nfilts)
                # compute output

                embedding = self.model(m_batch)
                output = self.classification_head(embedding)
                output = torch.sigmoid(output)
                
                output = output.view(bs, self.args.nb_seg, -1).mean(1)
                predictions = output.to('cpu').detach()

                
                l_predictions.extend(predictions)
                l_labels.extend(labels)
                l_paths.extend(path)

                # compute the loss
                labels = labels.to(self.args.device, non_blocking=True)
                loss = self.criterion['classification_loss'](output, labels)

                if isinstance(self.criterion['classification_loss'], torch.nn.CrossEntropyLoss):
                        loss = self.criterion['classification_loss'](output, torch.argmax(labels.long(), axis=1))
                else:
                    loss = self.criterion['classification_loss'](output, labels)

                l_losses.append(loss.to('cpu').detach())
                
        self._synchronize()

        l_predictions = all_gather(l_predictions)
        l_labels = all_gather(l_labels)
        l_losses = all_gather(l_losses)
        l_paths = all_gather(l_paths)
        
        prediction_dict = {}
        lable_dict = {}
        for i in range(len(l_paths)):
            prediction_dict[l_paths[i]] = l_predictions[i].tolist()
            lable_dict[l_paths[i]] = l_labels[i].tolist()
        
        return np.array(list(prediction_dict.values())), np.array(list(lable_dict.values())), np.mean(l_losses)
    
    def _synchronize(self):
        torch.cuda.empty_cache()
        dist.barrier()

