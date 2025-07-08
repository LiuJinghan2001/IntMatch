# This is the core code of Int*-Match
import sys, tqdm, soundfile
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
import time as tm

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, ema, tau, **kwargs):
        super(ECAPAModel, self).__init__()
        ## ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()
        self.speaker_loss = AAMsoftmax(m=m, s=s).cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(tm.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))
        self.p_cutoff = 0  #inter_class threshold
        self.cosinep_cutoff = 0  # intra_class threshold
        self.minp_cutoff = 1 / n_class
        self.ema = ema
        self.tau = tau
        self.cosine = 0
        self.maxclass = torch.ones((n_class,), dtype=torch.float).cuda() * -1.0
        self.flag = False
        self.changetime = 0
        self.test_step = test_step
        self.n_class = n_class

    def train_network(self, args, loader1, loader2):
        self.train()
        num = 1
        num_batch = 1
        index, top1, loss, loss1, loss2, quantity, indexulb = 0, 0, 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        EERs = []
        tstart = tm.time()

        for (idx, data, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s, y_ulb) in zip(loader1, loader2):
            self.scheduler.step()
            self.zero_grad()
            #labeled data process
            y_lb = torch.LongTensor(y_lb).cuda()
            y_ulb = torch.LongTensor(y_ulb).cuda()
            speaker_embedding, logits_lb = self.speaker_encoder.forward(data.cuda(), aug=True)

            logits_lb_mask = logits_lb.detach().clone()
            logits_lb_cosine = logits_lb.detach().clone()
            labels = y_lb.detach().clone()
            nloss1, prec = self.speaker_loss.forward(logits_lb.cuda(), mask=None, label=y_lb,
                                                     supervised=True)

            logits_lb_mask = torch.softmax(logits_lb_mask, dim=-1)
            max_probslb, max_idxlb = torch.max(logits_lb_mask, dim=-1)
            mask_lb = max_idxlb.eq(labels)

            one_hot = torch.zeros_like(logits_lb_cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1)
            # logits_lb_cosine = logits_lb_cosine * one_hot
            # cosinelb, _ = torch.max(logits_lb_cosine, dim=-1)
            cosinelb = torch.max(logits_lb_cosine * one_hot + -2.0 * (1 - one_hot), dim=-1)[0]
            unique_labels = labels.unique()
            for _, label in enumerate(unique_labels):
                if self.maxclass[label] == -1.0:
                    self.maxclass[label] = max(cosinelb[labels == label])
                else:
                    self.maxclass[label] = max(self.maxclass[label], max(cosinelb[labels == label]))

            if self.flag == False:
                nloss2 = 0.0
                loss2 += nloss2
                if mask_lb.sum() > 0:
                    if self.p_cutoff == 0:
                        self.p_cutoff = max_probslb[mask_lb == 1].mean()  #initialize inter-class threshold
                    else:
                        self.p_cutoff = self.ema * self.p_cutoff + (1 - self.ema) * max_probslb[mask_lb == 1].mean()
                if self.maxclass.mean() > self.tau:
                    self.flag = True
                    self.cosinep_cutoff = self.maxclass.mean()
            else:
                # unlabeled data process
                speaker_embedding1, logits_ulb_s = self.speaker_encoder.forward(x_ulb_s.cuda(), aug=True)
                speaker_embedding2, logits_ulb_w = self.speaker_encoder.forward(x_ulb_w.cuda(), aug=False)
                indexulb += len(y_ulb)
                logits_ulb_w = logits_ulb_w.detach()
                logits_ulb_w_cosine = logits_ulb_w.clone()

                logits_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
                max_probsulb_w, max_idxulb_w = torch.max(logits_ulb_w, dim=-1)

                mask = max_probsulb_w.ge(self.p_cutoff).float()
                pseudo_lb = max_idxulb_w.long()
                if mask.sum() == 0:
                    nloss2 = 0.0
                    loss2 += nloss2
                else:
                    nloss2, _ = self.speaker_loss.forward(logits_ulb_s.cuda(), mask=mask, label=pseudo_lb,
                                                          supervised=False)
                    loss2 += nloss2.detach().cpu().numpy()

                minprob = max_probsulb_w[mask == 0].sum()
                minindex = len(y_ulb) - mask.sum()
                cosine = torch.max(logits_ulb_w_cosine, dim=-1)[0]  # intra-class
                quantity += mask.sum()
                # quality += ((pseudo_lb.detach()).eq(y_ulb.detach()) * mask.detach()).sum()
                self.updateminp(minprob, minindex)  # unselected pls
                if mask.sum() != 0:
                    self.updateintra(cosine[mask == 1].mean())
                    self.updatethreshold(args, num, quantity / indexulb)

            nloss = nloss1 + nloss2
            nloss.backward()
            self.optim.step()
            time_used = tm.time() - tstart
            index += len(labels)
            top1 += prec
            loss1 += nloss1.detach().cpu().numpy()
            sys.stderr.write(tm.strftime("%H:%M:%S") + \
                             "[%7d]Lr:%5f,Lb:%.2f%%[%.2f],Ulb:%.2f%%" % (
                                 num, lr, 100 * (num / loader1.__len__()),
                                 time_used * loader1.__len__() / num / 60,
                                 100 * (num / loader2.__len__())) + \
                             "Ls: %.3f, Uls: %.3f, ACC: %2.2f%%, quantity: %.5f, p_cutoff: %.6f, cosinep_cutoff: %.4f, changetime: %d, cosineulb: %.4f, cosinelb: %.4f, minp_cutoff: %.6f\r" % (
                                 loss1 / num_batch, loss2 / num_batch, top1 / index * len(labels),
                                0 if indexulb == 0 else quantity / indexulb, self.p_cutoff, self.cosinep_cutoff,
                                 self.changetime, self.cosine, self.maxclass.mean(), self.minp_cutoff))
            sys.stderr.flush()

            if num % self.test_step == 0:
                sys.stdout.write("\n")
                self.save_parameters(args.model_save_path + "/model_%04d.model" % (num / self.test_step))
                EER, minDCF = self.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
                EERs.append(EER)
                if min(EERs) == EER:
                    self.save_parameters(args.model_save_path + "/model_%best.model")
                print(tm.strftime("%Y-%m-%d %H:%M:%S"),
                      "%d iterations, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%, minDCF%2.4f%%" % (
                          num, top1 / index * len(labels), EERs[-1], min(EERs), minDCF))
                score_file = open(args.score_save_path, "a+")
                score_file.write(
                    "%d iterations, LR: %f, Ls: %.3f, Uls: %.3f, ACC: %2.2f%%, EER: %2.2f%%, bestEER: %2.2f%%, minDCF: %2.4f%%, quantity: %.5f, p_cutoff: %.6f, cosinep_cutoff: %.4f, changetime: %d, cosineulb: %.4f, cosinelb: %.4f, minp_cutoff: %.6f\n" % (
                        num, lr, loss1 / num_batch, loss2 / num_batch, top1 / index * len(labels), EERs[-1], min(EERs), minDCF,
                        0 if indexulb == 0 else quantity / indexulb, self.p_cutoff, self.cosinep_cutoff, self.changetime,
                        self.cosine, self.maxclass.mean(), self.minp_cutoff))
                score_file.flush()
                lr = self.optim.param_groups[0]['lr']
                quantity = 0
                indexulb = 0
                index = 0
                top1 = 0
                loss1, loss2 = 0, 0
                num_batch = 0
            num += 1
            num_batch += 1


    @torch.no_grad()
    def updatethreshold(self, args, num, q):
        if self.cosine > self.cosinep_cutoff:
            self.p_cutoff = self.p_cutoff - (self.p_cutoff - self.minp_cutoff) * max(self.cosine, q)
            self.cosinep_cutoff = self.cosinep_cutoff + (self.maxclass.mean() - self.cosinep_cutoff) * max(self.cosine, q)


            score_file = open(args.score_save_path, "a+")
            score_file.write(
                "Change at %d iterations, newp_cutoff: %.6f, newcosinep_cutoff: %.6f, cosineulb: %.3f, quantity: %.4f\n" % (
                    num, self.p_cutoff, self.cosinep_cutoff, self.cosine, q))
            self.cosine = 0
            self.minp_cutoff = 1 / self.n_class
            self.changetime += 1
            score_file.flush()

    @torch.no_grad()
    def updateintra(self, cosine):
        if self.cosine == 0:
            self.cosine = cosine
        else:
            self.cosine = (1 - self.ema) * cosine + self.ema * self.cosine

    @torch.no_grad()
    def updateminp(self, minprob, minindex):
        if minindex > 0:
            self.minp_cutoff = (1 - self.ema) * minprob / minindex + self.ema * self.minp_cutoff

    def freeze_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def eval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(os.path.join(eval_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float64)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1, _ = self.speaker_encoder.forward(data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2, _ = self.speaker_encoder.forward(data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels = [], []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        self.train()
        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                # name = name.replace("Network.", "speaker_encoder.")
                if name not in self_state:
                    # print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

