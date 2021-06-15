import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn
from torch.utils.data import Dataset

class SmartsActionSequenceDataset(Dataset):
    def __init__(self, root='data', train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.tr_seq_length = 10
        self.tr_num_clips = 20

        self.data = []
        self.act_to_ix = {'loop': 0} #for now, only one scenario
                          
        images_dir = os.path.join(root, 'VideoVAE_classification', 'smarts_frames_aligned_by_video')
        ids = os.listdir(images_dir)
        
        self.id_to_ix = {identity:ix for ix, identity in enumerate(ids)}
        print(ids)

        self.ix_to_act = {ix:act for act, ix in self.act_to_ix.items()}
        self.ix_to_id = {ix:identity for identity, ix in self.id_to_ix.items()}


        # loop through people
        for identity in ids:
            actions = sorted(os.listdir(os.path.join(images_dir, identity)))

            # loop though action
            for act in actions:
                frames = sorted(os.listdir(os.path.join(images_dir, identity, act)))

                num_train = len(frames) * 2 // 3

                #######################################################
                #                   Training set                      #
                #######################################################
                if self.train:
                    start = 0
                    end = num_train

                    # Can sample just 20 mini-clips of length-10 from each video here
                    # by uncommenting the following line.
                    #sample_ix = np.random.choice(range(start, end-10), 20)
                    sample_ix = range(start, end-self.tr_seq_length)

                    for ix, seq_start in enumerate(sample_ix):
                        seq_i = []
                        for t in range(self.tr_seq_length):
                            fname = frames[seq_start + t]

                            # get frame path
                            # NOTE: fname should be juse number
                            f_path = os.path.join(images_dir, identity, act, fname)

                            seq_i.append({
                                'img_path': f_path,
                                'action_label': self.act_to_ix[act],
                                'identity_label': self.id_to_ix[identity],
                                'action': act,
                                'identity': identity,
                            })

                        self.data.append(seq_i)
                #######################################################
                #                    Testing set                      #
                #######################################################
                else:
                    start = num_train
                    end = len(frames)

                    # NOTE: No need to form a sequence here, I think.
                    for ix in range(start, end):
                        fname = frames[ix]
                        f_path = os.path.join(images_dir, identity, act, fname)

                        self.data.append({
                            'img_path': f_path,
                            'action_label': self.act_to_ix[act],
                            'identity_label': self.id_to_ix[identity],
                            'action': act,
                            'identity': identity,
                        })

    def __getitem__(self, idx):
        train = self.train
        tr_seq_length = self.tr_seq_length
        tr_num_clips = self.tr_num_clips
        data = self.data[idx]

        if self.train:
            img_seq = []
            act_label_seq = []
            id_label_seq = []

            # NOTE: here
            for t in range(tr_seq_length):
                data_t = data[t]
                img_path = data_t['img_path']
                action_label = data_t['action_label']     
                identity_label = data_t['identity_label'] 
                action = data_t['action']
                identity = data_t['identity']
                im = Image.open(img_path)
            
                if self.transform:
                    im = self.transform(im)

                img_seq.append(im)
                act_label_seq.append(action_label)
                id_label_seq.append(identity_label)
            
            img_seq = torch.stack(img_seq, dim=0)
            act_label_seq = torch.tensor(act_label_seq)
            id_label_seq = torch.tensor(id_label_seq)
                
            return img_seq, act_label_seq, id_label_seq, action, identity
        
        else:
            # testing
            img_path = data['img_path']
            action_label = data['action_label']
            identity_label = data['identity_label']
            action = data['action']
            identity = data['identity']
            im = Image.open(img_path)

            if self.transform:
                im = self.transform(im)

            return im, action_label, identity_label, action, identity

    def __len__(self):
        return len(self.data)


class SmartsActionClassificationDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.tr_seq_length = 10
        self.tr_num_clips = 20

        self.data = []
        self.act_to_ix = {'loop': 0} #for now, only one scenario
                          
        images_dir = os.path.join(root, 'VideoVAE_classification', 'smarts_frames_aligned_by_video')
        ids = os.listdir(images_dir)
        
        self.id_to_ix = {identity:ix for ix, identity in enumerate(ids)}


        # loop through people
        for identity in ids:
            actions = sorted(os.listdir(os.path.join(images_dir, identity)))

            # loop though action
            for act in actions:
                frames = sorted(os.listdir(os.path.join(images_dir, identity, act)))

                num_train = len(frames) * 2 // 3

                if self.train:
                    # Can randomly sample 20% of data here.
                    start = 0
                    end = num_train
                else:
                    start = num_train
                    end = len(frames)

                for ix in range(start, end):
                    fname = frames[ix]

                    f_path = os.path.join(images_dir, identity, act, fname)

                    self.data.append({
                            'img_path': f_path,
                            'action_label': self.act_to_ix[act],
                            'identity_label': self.id_to_ix[identity],
                            'action': act,
                            'identity': identity,
                            })

    def __getitem__(self, idx):
        train = self.train
        tr_seq_length = self.tr_seq_length
        tr_num_clips = self.tr_num_clips
        data = self.data[idx]

        img_path = data['img_path']
        action_label = data['action_label']
        identity_label = data['identity_label']
        action = data['action']
        identity = data['identity']
        im = Image.open(img_path)

        if self.transform:
            im = self.transform(im)

        return im, action_label, identity_label, action, identity

    def __len__(self):
        return len(self.data)