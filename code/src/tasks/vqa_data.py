# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv
import os
import csv
#import sys
#sys.path.insert(1, '../')
# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'

from lxrt.SlowFast.slowfast.config.defaults import get_cfg
from lxrt.SlowFast.slowfast.datasets.tgif_direct import TGIF



cfg = get_cfg()
cfg_file = "src/lxrt/SlowFast/configs/Kinetics/c2/SLOWFAST_8x8_R50.yaml"
cfg.merge_from_file(cfg_file)

def assert_exists(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)

class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/vqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        
    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

'''API for TGIF dataset which loads GIF tensor and prepares question and 
   options per GIF
   Repo looks like : 
   /lxmert/data/tgif/dataframe -> This contains GIF tensors and csv file
        /lxmert/data/tgif/dataframe/gif_tensors -> Contains GIF tensors

   /lxmert/data/tgif/vocabulary -> This contains vocabulary, word2idx, ans2idx etc.
        /lxmert/data/tgif/vocabulary/word_matrix_<data_type>.pkl
        /lxmert/data/tgif/vocabulary/word_to_index_<data_type>.pkl
        /lxmert/data/tgif/vocabulary/index_to_word_<data_type>.pkl
        /lxmert/data/tgif/vocabulary/ans_to_index_<data_type>.pkl
        /lxmert/data/tgif/vocabulary/index_to_ans_<data_type>.pkl
'''
class TGIFDataset(Dataset):
    def __init__(self, dataset_name='train', data_type=None, dataframe_dir=None, vocab_dir=None):
        self.dataframe_dir = dataframe_dir # of the form data/tgif/vocabulary
        self.vocab_dir = vocab_dir # of the form data/tgif/dataframe
        self.data_type = data_type # 'TRANS'
        self.dataset_name = dataset_name # 'train' or 'val' or 'test'

        self.csv = self.read_from_csvfile()
        self.header2idx = self.header2idx()
        self.gif_names = self.csv[:,self.header2idx['gif_name']]
        self.gif_tensor = None
        self.questions = self.csv[:,self.header2idx['question']]
        self.answers = self.csv[:,self.header2idx['answer']]
        self.mc_options = self.csv[:,self.header2idx['a1']:header2idx['a5']+1]
        ## GIF LOADER ##
        ## NOTE: May have to change the relative path of gif dir as 
        ## an extra argument to TGIF class init
        loader  = TGIF(cfg, "train")
        self.get_gif_tensor = loader.__getitem__
        
    def __getitem__(self, i): # whats the argument for this
        gif_path = os.path.join(self.dataframe_dir, 'gif_tensors')
        #pick up ith gif_tensor
        #NOTE: gif_path is only the gif name, not the relative path
        # REturn value: tuple (slow frames, fast frames) where frame -> (t, 3, h, w)
        gif_tensor = self.get_gif_tensor(gif_path)
        return self.gif_tensor, self.questions[i], self.mc_options[i], self.answers[i]

    def header2idx(self):
        return {'gif_name':0,'question':1,'a1':2,'a2':3,'a3':4,'a4':5,'a5':6,'answer':7,'vid_id':8,'key':9}

    def read_from_csvfile(self):
        assert self.data_type in ['TRANS', 'ACTION'] # ACTION just for starting, will be using TRANS finally

        self.total_q=[]
        if self.data_type=='TRANS':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_transition_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_transition_question.csv')

            
            with open(os.path.join(self.dataframe_dir, 'Total_transition_question.csv')) as file:
                csv_reader = csv.reader(file, delimiter='\t')
                for row in csv_reader:
                    self.total_q.append(row)

        elif self.data_type=='ACTION':
            train_data_path = os.path.join(self.dataframe_dir, 'Train_action_question.csv')
            test_data_path = os.path.join(self.dataframe_dir, 'Test_action_question.csv')

            with open(os.path.join(self.dataframe_dir, 'Total_action_question.csv')) as file:
                csv_reader = csv.reader(file, delimiter='\t')
                for row in csv_reader:
                    self.total_q.append(row)
        self.total_q.pop(0)

        assert_exists(train_data_path)
        assert_exits(test_data_path)

        csv_data=[]
        if self.dataset_name=='train':
            with open(train_data_path) as file:
                csv_reader = csv.reader(file, delimiter='\t')
                for row in csv_reader:
                    csv_data.append(row)
        elif self.dataset_name=='test':
            with open(test_data_path) as file:
                csv_reader = csv.reader(file, delimiter='\t')
                for row in csv_reader:
                    csv_data.append(row)
        csv_data.pop(0)

        return np.asarray(csv_data)
    '''
    def build_vocabulary(self):

    def load_vocabulary(self):
        word_matrix_path = os.path.join(self.vocab_dir, 'word_matrix_%s.pkl'%self.data_type)
        word2idx_path = os.path.join(self.vocab_dir, 'word_to_index_%s.pkl'%self.data_type)
        idx2word_path = os.path.join(self.vocab_dir, 'index_to_word_%s.pkl'%self.data_type)
        ans2idx_path = os.path.join(self.vocab_dir, 'ans_to_index_%s.pkl'%self.data_type)
        idx2ans_path = os.path.join(self.vocab_dir, 'index_to_ans_%s.pkl'%self.data_type)

        if not os.path.exists(word_matrix_path) and os.path.exists(word2idx_path) and \
                os.path.exists(idx2word_path) and os.path.exists(ans2idx_path) and \
                os.path.exists(idx2ans_path):
                build_vocabulary()
    '''




"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""

class FrameQADataset(object):
    def __init__(self, dataset_name='train', data_type=None, dataframe_dir=None, vocab_dir=None, category ="frameqa" ):
        self.dataframe_dir = dataframe_dir # of the form data/tgif/vocabulary
        self.vocab_dir = vocab_dir # of the form data/tgif/dataframe
        self.data_type = data_type # 'TRANS'
        self.dataset_name = dataset_name # 'train' or 'val' or 'test'

        self.csv, all_data = self.read_from_csvfile(category)
        self.header2idx = self.header2idx()
        self.gif_names = self.csv[:,self.header2idx['gif_name']]
        self.gif_tensor = None
        self.questions = self.csv[:,self.header2idx['question']]
        self.answer = self.csv[:,self.header2idx['answer']]
        self._build_ans_vocab(all_data[:,self.header2idx['answer']])
        ## GIF LOADER ##
        ## NOTE: May have to change the relative path of gif dir as 
        ## an extra argument to TGIF class init
        self.root_path = "/users/cdwivedi/RL_EXP/IDL/project/tgif-qa/code/dataset/tgif/frame_gifs/"+dataset_name+"/"
        loader  = TGIF(cfg, dataset_name,root_path=self.root_path )
        self.get_gif_tensor = loader.__getitem__
        self.check_gif = loader.check_gif
        
    def _build_ans_vocab(self, all_answers):
        vocab = set()
        for ans in all_answers:
            vocab.add(str(ans))
        self.vocab = sorted(list(vocab))
        self.id2ans = self.vocab
        self.ans2id = dict(zip(self.vocab, np.arange(len(self.vocab))))
        self.vocab_len = len(self.vocab)
        self.num_answers = self.vocab_len
        self.label2ans = self.id2ans
        
    def get_one_hot(self, i):
        vec = np.zeros(self.vocab_len)
        vec[i] = 1
        return vec
        
    def __getitem__(self, i): # whats the argument for this
        gif_path = self.gif_names[i]
        #pick up ith gif_tensor
        #NOTE: gif_path is only the gif name, not the relative path
        # REturn value: tuple (slow frames, fast frames) where frame -> (t, 3, h, w)
        patience = 10
        counter = 0
        while(counter < patience):
            if self.check_gif(gif_path):
                gif_tensor = self.get_gif_tensor(gif_path)
                break
            else:
                counter += 1
                i = np.random.choice(self.__len__())
                gif_path = self.gif_names[i]
        return gif_tensor[0], gif_tensor[1], self.questions[i], self.get_one_hot(self.ans2id[self.answer[i]])

    def __len__(self):
        return len(self.questions)//2
    
    def header2idx(self):
        return {'gif_name':0,'question':1,'answer':2}

    def read_from_csvfile(self, category=None):
        print(category)
        train_data_path = os.path.join(self.dataframe_dir, 'Train_'+category+'_question.csv')
        test_data_path = os.path.join(self.dataframe_dir, 'Test_'+category+'_question.csv')
        total_data_path = os.path.join(self.dataframe_dir, 'Total_'+category+'_question.csv')
        csv_data=[]
        if self.dataset_name=='train':
            with open(train_data_path) as file:
                csv_reader = csv.reader(file, delimiter='\t')
                for row in csv_reader:
                    csv_data.append(row)
        elif self.dataset_name=='test':
            with open(test_data_path) as file:
                csv_reader = csv.reader(file, delimiter='\t')
                for row in csv_reader:
                    csv_data.append(row)
        csv_data.pop(0)
        total_csv_data=[]
        with open(total_data_path) as file:
            csv_reader = csv.reader(file, delimiter='\t')
            for row in csv_reader:
                total_csv_data.append(row)

        total_csv_data.pop(0)
        return np.asarray(csv_data), np.asarray(total_csv_data)


class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        if 'train' in dataset.splits:
            img_data.extend(load_obj_tsv('data/mscoco_imgfeat/train2014_obj36.tsv', topk=topk))
        if 'valid' in dataset.splits:
            img_data.extend(load_obj_tsv('data/mscoco_imgfeat/val2014_obj36.tsv', topk=topk))
        if 'minival' in dataset.splits:
            # minival is 5K images in the intersection of MSCOCO valid and VG,
            # which is used in evaluating LXMERT pretraining performance.
            # It is saved as the top 5K features in val2014_obj36.tsv
            if topk is None:
                topk = 5000
            img_data.extend(load_obj_tsv('data/mscoco_imgfeat/val2014_obj36.tsv', topk=topk))
        if 'nominival' in dataset.splits:
            # nominival = mscoco val - minival
            img_data.extend(load_obj_tsv('data/mscoco_imgfeat/val2014_obj36.tsv', topk=topk))
        if 'test' in dataset.name:      # If dataset contains any test split
            img_data.extend(load_obj_tsv('data/mscoco_imgfeat/test2015_obj36.tsv', topk=topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


