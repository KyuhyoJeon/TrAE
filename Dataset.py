import pandas as pd
import numpy as np
import torch
import pickle

from torch.utils.data import Dataset, DataLoader


'''
needs to be coded for getting other types of trajectory
1. normal types
2. mixed types

The ratio of train/valid/test is 7/1/2 but 3/1/1 in paper
'''


class Trajectory3Dnewset(Dataset):
    def __init__(self, path='Data/3D/new/csv/', size=None, model_type='A', data_type='normal', flag='train', ratio = [7,1,2]):
        # size [seq_len, label_len, control_num]
        # info
        if size == None:
            if model_type == 'A':
                self.seq_len = 2656
                self.class_num = 5
            else:
                self.seq_len = 400
                self.label_len = 100
                self.pred_len = 100
        else:
            if model_type == 'A':
                self.seq_len = size[0]
                self.class_num = size[1]
            else:
                self.seq_len = size[0]
                self.label_len = size[1]
                self.pred_len = size[2]
        self.model_type = model_type
        self.data_type = data_type
        self.flag = flag
        self.path = path
        self.ratio = ratio
        
        self.__read_data__()

    def __read_data__(self):
        '''
        X = ['Rmi_1', 'Rmi_2', 'Rmi_3', 
            'Vmi_1', 'Vmi_2', 'Vmi_3', 
            'Wmb_1', 'Wmb_2', 'Wmb_3', 
            'Accm_1', 'Accm_2', 'Accm_3', 
            'angEuler_1', 'angEuler_2', 'angEuler_3', 
            'FinOut_1', 'FinOut_2', 'FinOut_3', 'FinOut_4', 
            'FinCmd_1', 'FinCmd_2', 'FinCmd_3', 'FinCmd_4']
        Y = [Class]
        '''
        NSamples = 500 if self.model_type == 'A' else 100
        
        d_type = {'train':0, 'train_c':0, 'valid':1, 'test':2, 'pred':3}
        lefts = [0, 
                 int(10*(sum(self.ratio[:1])/sum(self.ratio))), 
                 int(10*(sum(self.ratio[:2])/sum(self.ratio))), 
                 0]
        rights = [int(10*(sum(self.ratio[:1])/sum(self.ratio))), 
                  int(10*(sum(self.ratio[:2])/sum(self.ratio))), 
                  10, 
                  10]
        
        data_x_raw = []
        data_y_raw = []
        
        
        for i, tr_type in enumerate(['Normal', 'Burntime', 'Xcpposition', "ThrustTiltAngle", 'Finbias']):
            left = lefts[d_type[self.flag]]
            right = rights[d_type[self.flag]]
                
            for tr_i in range(NSamples):
                if tr_i%10 < left or tr_i%10 >= right:
                    continue
                tr = pd.read_csv(self.path + f'Type_{i+1}_{tr_i+1}.csv', header=None)
                tr.columns = ['Rmi_1', 'Rmi_2', 'Rmi_3', 
                              'Vmi_1', 'Vmi_2', 'Vmi_3', 
                              'Wmb_1', 'Wmb_2', 'Wmb_3', 
                              'Accm_1', 'Accm_2', 'Accm_3', 
                              'angEuler_1', 'angEuler_2', 'angEuler_3', 
                              'FinOut_1', 'FinOut_2', 'FinOut_3', 'FinOut_4', 
                              'err_FinBias_1', 'err_FinBias_2', 'err_FinBias_3', 'err_FinBias_4', 
                              'FinCmd_1', 'FinCmd_2', 'FinCmd_3', 'FinCmd_4', 
                              'err_BurnTime', 'err_Tilt_1', 'err_Tilt_2', 'err_delXcp']
                tr = tr[['Rmi_1', 'Rmi_2', 'Rmi_3', 
                         'Vmi_1', 'Vmi_2', 'Vmi_3', 
                         'Wmb_1', 'Wmb_2', 'Wmb_3', 
                         'Accm_1', 'Accm_2', 'Accm_3', 
                         'angEuler_1', 'angEuler_2', 'angEuler_3', 
                         'FinOut_1', 'FinOut_2', 'FinOut_3', 'FinOut_4', 
                         'FinCmd_1', 'FinCmd_2', 'FinCmd_3', 'FinCmd_4']].to_numpy()
                tr[0] = 4000 - tr[0]
                if self.model_type == 'A':
                    terminal = tr[-1].copy()
                    # terminal[3]=terminal[4]=terminal[5]=terminal[6]=terminal[7]=terminal[8]=terminal[12]=terminal[13]=terminal[14]=terminal[19]=terminal[20]=terminal[21]=terminal[22]=0
                    tr = np.concatenate((tr, np.tile(terminal, (self.seq_len-len(tr), 1))), axis=0)
                    if i == 0: 
                        N = int(NSamples*5*0.7) if self.data_type == 'normal' else int(NSamples*0.7)
                        data_x_raw = [tr for _ in range(N)]
                        data_y_raw = [i for _ in range(N)]
                    else:
                        data_x_raw.append(tr)
                        data_y_raw.append(i)
                else:
                    for idx in range(0, len(tr)-self.seq_len-self.pred_len+1, 1):
                        s_begin = idx
                        s_end = s_begin + self.seq_len
                        r_begin = s_end - self.label_len
                        r_end = r_begin + self.label_len + self.pred_len
                        
                        data_x_raw.append(tr[s_begin:s_end])
                        data_y_raw.append(tr[r_begin:r_end])
                if i == 0:
                    break
            if self.data_type == 'normal' and self.flag == 'train':
                break
        
        self.data_x = np.array(data_x_raw)
        self.data_y = np.array(data_y_raw)

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]

        return x, y

    def __len__(self):
        return len(self.data_x)
    
    
class Trajectory2Dset(Dataset):
    def __init__(self, path='Data/2D/', size=None, model_type='A', data_type='normal', flag='train', ratio=[7,2,1]):
        # size [seq_len, label_len, control_num]
        # info
        if size == None:
            if model_type == 'A':
                self.seq_len = 2144
                self.class_num = 4
            else:
                self.seq_len = 400
                self.label_len = 100
                self.pred_len = 100
        else:
            if model_type == 'A':
                self.seq_len = size[0]
                self.class_num = size[1]
            else:
                self.seq_len = size[0]
                self.label_len = size[1]
                self.pred_len = size[2]
        self.model_type = model_type
        self.data_type = data_type
        self.flag = flag
        self.path = path
        self.ratio = ratio
        
        self.__read_data__()

    def __read_data__(self):
        '''
        Trajectory types: Normal, autopilot_lag, LOSrate_bias, LOSrate_delay
        Target distance = uniform(4500, 5500)
        states = [x, y, vm, path_angle] + 1 control input, 5 states
        label = 0, 1, 2, 3 for normal, autopilot_lag, LOSrate_bias, LOSrate_delay respectively, 4 classes
        minimum length of trajectory = 1360 (1360*0.01 = 13.6 seconds)
        maximum length of trajectory = 2131 (2131*0.01 = 21.31 seconds)
        fixed sequence length = 2144 (2400*0.01 = 24 seconds)
        '''
        with open(self.path+'Trajectories.pickle', 'rb') as f:
            Dataset = pickle.load(f)
        
        NSamples = 500 if self.model_type == 'A' else 100
        if self.data_type == 'normal' and self.flag == 'train':
            NSamples = NSamples*self.class_num
        
        d_type = {'train':0, 'train_c':0, 'valid':1, 'test':2, 'pred':3}
        lefts = [0, 
                 int(10*(sum(self.ratio[:1])/sum(self.ratio))), 
                 int(10*(sum(self.ratio[:2])/sum(self.ratio))), 0]
        rights = [int(10*(sum(self.ratio[:1])/sum(self.ratio))), 
                  int(10*(sum(self.ratio[:2])/sum(self.ratio))), 
                  10, 10]
        
        data_x_raw = []
        data_y_raw = []
        
        for idx, tr_type in enumerate(Dataset):
            states = Dataset[tr_type]['states']
            inputs = Dataset[tr_type]['actions']
            
            left = lefts[d_type[self.flag]]
            right = rights[d_type[self.flag]]
            
            for i in range(NSamples):
                if i%10 < left or i%10 >= right:
                    continue
                tr = np.concatenate((states[i], np.insert(inputs[i], 0, 0).reshape(-1 ,1)), axis=1)
                if self.model_type == 'A':
                    terminal = tr[-1].copy()
                    # terminal[2]=terminal[4]=0
                    tr = np.concatenate((tr, np.tile(terminal, (self.seq_len-len(tr), 1))), axis=0)
                    data_x_raw.append(tr)
                    data_y_raw.append(idx)
                else:
                    for idx in range(0, len(tr)-self.seq_len-self.pred_len+1, 1):
                        s_begin = idx
                        s_end = s_begin + self.seq_len
                        r_begin = s_end - self.label_len
                        r_end = r_begin + self.label_len + self.pred_len
                        
                        data_x_raw.append(tr[s_begin:s_end])
                        data_y_raw.append(tr[r_begin:r_end])
                        
            if self.data_type == 'normal' and self.flag == 'train':
                break
            
        self.data_x = np.array(data_x_raw)
        self.data_y = np.array(data_y_raw)

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]

        return x, y

    def __len__(self):
        return len(self.data_x)


def data_provider(args, flag):
    shuffle_flag = True
    drop_last = False
    batch_size = args.batch_size
    
    if args.model == 'A':
        size=[args.seq_len, args.class_num]
    elif args.model == 'B':
        size=[args.seq_len, args.label_len, args.pred_len]
    
    path = args.root_path + args.data + args.data_path
    
    Data = Trajectory3Dnewset if args.data == '3D/new/' else Trajectory2Dset #if args.data == '2D/' else Trajectory3Doldset
    
    data_set = Data(
        path=path,
        size=size,
        model_type = args.model,
        data_type = args.data_type,
        flag = flag,
        ratio= args.ratio)
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
        
    return data_set, data_loader


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size, 4)
    subsequent_mask = torch.zeros(attn_shape, dtype=torch.float32)
    for i in range(size):
        for j in range(i+1, size):
            subsequent_mask[0][i][j]=1
    return subsequent_mask == 0


class Mask:
    def __init__(self, configs, src, tgt):
        self.seq_len = src.size(1)
        self.input_channel = configs.enc_in
        self.output_channel = configs.control_num
        self.set_src_mask(src)
        self.set_tgt_mask(tgt)

    def set_src_mask(self, src):
        self.src_enc_mask = torch.ones(1, self.seq_len, self.seq_len, self.input_channel).type_as(src)
        self.src_dec_mask = torch.ones(1, self.seq_len, self.seq_len, self.output_channel).type_as(src)
        
    def set_tgt_mask(self, tgt):
        self.tgt_mask = subsequent_mask(tgt.size(1)).type_as(tgt.data)
        
    def get_src_mask(self):
        return self.src_enc_mask, self.src_dec_mask
    
    def get_tgt_mask(self):
        return self.tgt_mask
    
    def update_tgt_mask(self, tgt):
        self.set_tgt_mask(tgt)
        return self.tgt_mask
    
    
