# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator, FrameQADataset

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
from logger_utils import logger as log
logger = log("WithAttention") 

# def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
#     dset = 7(splits)
#     tset = VQATorchDataset(dset)
#     evaluator = VQAEvaluator(dset)
#     data_loader = DataLoader(
#         tset, batch_size=bs,
#         shuffle=shuffle, num_workers=args.num_workers,
#         drop_last=drop_last, pin_memory=True
#     )

#     return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

def get_data_tuple(args_train, bs=32,shuffle=False, drop_last=False, dataset_name="test") -> DataTuple:
    dset = FrameQADataset(dataframe_dir="../../tgif-qa/dataset/", \
                          dataset_name=dataset_name)
    num_workers = 8
    data_loader = DataLoader(
        dset, batch_size=bs,
        shuffle=shuffle, num_workers=num_workers,
        drop_last=drop_last, pin_memory=True
    )
    return DataTuple(dataset=dset, loader=data_loader, evaluator=None)


class VQA:
    def __init__(self, attention=False):
        # Datasets
        print("Fetching data")
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True, dataset_name ="test"
        )
        print("Got data")
        print("fetching val data")
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=args.batch_size,
                shuffle=False, drop_last=False, dataset_name="test"
            )
            print("got data")
        else:
            self.valid_tuple = None
        print("Got data")
        
        # Model
        print("Making model")
        self.model = VQAModel(self.train_tuple.dataset.num_answers, attention)
        print("Ready model")
        # Print model info:
        print("Num of answers:")
        print(self.train_tuple.dataset.num_answers)
        # print("Model info:")
        # print(self.model)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        log_freq = 810
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        flag = True
        for epoch in range(args.epochs):
            quesid2ans = {}
            correct = 0
            total_loss = 0
            total = 0
            print("Len of the dataloader: ", len(loader))
#             Our new TGIFQA-Dataset returns:
#             return gif_tensor, self.questions[i], self.ans2id[self.answer[i]]
            for i, (feats1, feats2, sent, target) in iter_wrapper(enumerate(loader)):
                ques_id, boxes = -1, None
                self.model.train()
                self.optim.zero_grad()

                
                feats1, feats2, target = feats1.cuda(), feats2.cuda(), target.cuda()
                feats = [feats1, feats2]
                
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)
        
                total_loss += loss.item()
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                score_t, target = target.max(1)
                correct += (label == target).sum().cpu().numpy()
                total += len(label)
                #if epoch > -1:
                    #for l,s,t in zip(label, sent, target):
                    #    print(l)
                    #    print(s)
                    #    print("Prediction", loader.dataset.label2ans[int(l.cpu().numpy())])
                    #    print("Answer", loader.dataset.label2ans[int(t.cpu().numpy())])
                
                if  i % log_freq == 1 and i > 1:
                    results=[]
                    for l,s,t in zip(label, sent, target):
                        result = []
                        result.append(s)
                        result.append("Prediction: {}".format(loader.dataset.label2ans[int(l.cpu().numpy())]))
                        result.append("Answer: {}".format(loader.dataset.label2ans[int(t.cpu().numpy())]))
                        results.append(result)
                        torch.cuda.empty_cache()
                    val_loss, val_acc, val_results = self.val(eval_tuple)
                    logger.log(total_loss/total, correct/total*100, val_loss, val_acc, epoch, results,val_results )
                    
            print("=="*30)
            print("Accuracy = " , correct/total*100)
            print("Loss =" , total_loss/total)
            print("=="*30)
#             log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

#             if self.valid_tuple is not None:  # Do Validation
#                 valid_score = self.evaluate(eval_tuple)
#                 if valid_score > best_valid:
#                     best_valid = valid_score
#                     self.save("BEST")

#                 log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
#                            "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

#             print(log_str, end='')

#             with open(self.output + "/log.log", 'a') as f:
#                 f.write(log_str)
#                 f.flush()

            self.save("Check"+str(epoch))
    def val(self, eval_tuple):
        dset, loader, evaluator = eval_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        self.model.eval()
        best_valid = 0.
        flag = True
        quesid2ans = {}
        correct = 0
        total_loss = 0
        total = 0
        results= []
        print("Len of the dataloader: ", len(loader))
#             Our new TGIFQA-Dataset returns:
#             return gif_tensor, self.questions[i], self.ans2id[self.answer[i]]
        with torch.no_grad():
            for i, (feats1, feats2, sent, target) in iter_wrapper(enumerate(loader)):
                ques_id, boxes = -1, None


                feats1, feats2, target = feats1.cuda(), feats2.cuda(), target.cuda()
                feats = [feats1, feats2]

                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                total_loss += loss.item()


                score, label = logit.max(1)
                score_t, target = target.max(1)
                correct += (label == target).sum().cpu().numpy()
                total += len(label)
                for l,s,t in zip(label, sent, target):
                    result = []
                    result.append(s)
                    result.append("Prediction: {}".format(loader.dataset.label2ans[int(l.cpu().numpy())]))
                    result.append("Answer: {}".format(loader.dataset.label2ans[int(t.cpu().numpy())]))
                    results.append(result)
            return total_loss/total, correct/total*100, results
        
    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    print("Inside main")
    # Build Class
    vqa = VQA(attention=True)
    
    #rest of the model
    for param in vqa.model.parameters():
        param.requires_grad = False
        
    # to train cross model encoder along with cnn_bridge
    for param in vqa.model.lxrt_encoder.model.bert.encoder.x_layers.parameters():
        param.requires_grad = True

    for param in vqa.model.lxrt_encoder.model.bert.encoder.r_layers.cnn_bridge.parameters():
        param.requires_grad = True
    for param in vqa.model.lxrt_encoder.model.bert.encoder.r_layers.s5.parameters():
        param.requires_grad = True
    for param in vqa.model.logit_fc.parameters():
        param.requires_grad = True
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        print("Inside test loop")
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('minival', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print("Inside train condition")
#         print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
#             print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            pass
#             print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)


