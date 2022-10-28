from tqdm import tqdm
import torch
import numpy as np
import torch.distributed as dist
from torch.cuda.amp import autocast,GradScaler
from utils.ddp_util import all_gather
import utils.metric as metric
import torch.nn.functional as F

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
    evaluation_set = None
    evaluation_loader = None
    spec = None

    def run(self):
        
        self.best_eer = 99.
        self.best_min_dcf1 = 99.
        self.best_min_dcf10 = 99.

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
                
        
                description = '%s epoch: %d '%(self.args.name, epoch)

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
                            self.logger.log_metric('loss', _loss, step = idx_ct_start+idx)
                            self.logger.log_metric('lr', lr, step = idx_ct_start+idx)
                            _loss = 0.
                            
        self.lr_scheduler.step()        

    def test(self, epoch, mode, dataloader):
        with self.model.no_sync():
            embeddings = self._evaluate(dataloader) 

        if self.args.flag_parent:

            labels = []
            cos_sims = []

            for item in tqdm(self.dataset.test_trials, desc='test', ncols=150):
                cos_sims.append(self._calculate_cosine_similarity(embeddings[item.key1], embeddings[item.key2]))
                labels.append(int(item.label))

            eer = metric.calculate_EER(
                scores=cos_sims, labels=labels
            )
            min_dcf1 = metric.calculate_MinDCF(
                scores=cos_sims, labels=labels, p_target=0.01, c_miss=1, c_false_alarm=1
            )
            min_dcf10 = metric.calculate_MinDCF(
                scores=cos_sims, labels=labels, p_target=0.01, c_miss=10, c_false_alarm=1
            )            
            
            self.logger.log_metric('EER', eer, epoch_step=epoch)
            self.logger.log_metric('Min_DCF1', min_dcf1, epoch_step=epoch)
            self.logger.log_metric('Min_DCF10', min_dcf10, epoch_step=epoch)

            
            if eer < self.best_eer:
                self.best_eer = eer
                self.logger.log_metric('Best_EER', eer, epoch_step=epoch)

            if min_dcf1 < self.best_min_dcf1:
                self.best_min_dcf1 = min_dcf1
                self.logger.log_metric('Best_Min_DCF1', min_dcf1, epoch_step=epoch)

            if min_dcf10 < self.best_min_dcf10:
                self.best_min_dcf10 = min_dcf10
                self.logger.log_metric('Best_Min_DCF10', min_dcf10, epoch_step=epoch)

    def _evaluate(self, dataloader):
        self.model.eval()
        self.classification_head.eval()
        
        l_embeddings = []
        l_paths = []

        with torch.set_grad_enabled(False):
            for m_batch, labels, path in tqdm(dataloader, ncols=120):
                m_batch = m_batch.to(self.args.device, non_blocking=True)
                bs = m_batch.shape[0]
                m_batch = m_batch.view(-1, self.args.frame_length, self.args.nfilts)
                # compute output

                embedding = self.model(m_batch)
                embedding = self.classification_head(embedding, is_test=True).to('cpu')
                embedding = embedding.view(bs, self.args.nb_seg, -1)
                
                l_embeddings.extend(embedding)
                l_paths.extend(path)

                
        self._synchronize()

        l_embeddings = all_gather(l_embeddings)
        l_paths = all_gather(l_paths)
        
        embedding_dict = {}
        for i in range(len(l_paths)):
            
            embedding_dict['/'.join(l_paths[i].split('/')[4:])] = l_embeddings[i]
        
        return embedding_dict
        
    def _calculate_cosine_similarity(self, trials, enrollments):
        
        buffer1 = F.normalize(trials, p=2, dim=-1)
        buffer2 = F.normalize(enrollments, p=2, dim=-1)

        dist = F.pairwise_distance(buffer1.unsqueeze(-1), buffer2.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()

        score = -1 * np.mean(dist)

        return score

    def _synchronize(self):
        torch.cuda.empty_cache()
        dist.barrier()

