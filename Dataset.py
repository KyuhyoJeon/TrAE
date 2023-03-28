import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class ClassifySet(Dataset):
    def __init__(self, path='Data/3D/', size=None, input='x_new.npy', output='y_new.npy', model_type='A', flag='train'):
        # size [seq_len, label_len, control_num]
        # info
        if size == None:
            if model_type == 'A':
                self.seq_len = 2656
                self.class_num = 5
            else:
                self.seq_len = 1992
                self.label_len = 664
                self.pred_len = 664
        else:
            if model_type == 'A':
                self.seq_len = size[0]
                self.class_num = size[1]
            else:
                self.seq_len = size[0]
                self.label_len = size[1]
                self.pred_len = size[2]
        self.model_type = model_type
        self.flag = flag
        
        # init
        self.path_x = path+input # 
        self.path_y = path+output # 
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
        data_x_raw = []
        data_y_raw = []
        tr = {}
        for i, tr_type in enumerate(['Normal', 'Burntime', 'Xcpposition', "ThrustTiltAngle", 'Finbias']):
            if i == 0:
                left = 0
                right = 1
            else:
                d_type = {'train':0, 'valid':1, 'test':2, 'pred':3}
                lefts = [0, int(500*0.7), int(500*0.8), 0]
                right = [int(500*0.7), int(500*0.8), 500, 500]
                left = lefts[d_type[self.flag]]
                right = right[d_type[self.flag]]
            for tr_i in range(left, right):
                tr = pd.read_csv(self.path_x + f'Type_{i+1}_{tr_i+1}.csv', header=None)
                tr.columns = ['Rmi_1', 'Rmi_2', 'Rmi_3', 'Vmi_1', 'Vmi_2', 'Vmi_3', 'Wmb_1', 'Wmb_2', 'Wmb_3', 'Accm_1', 'Accm_2', 'Accm_3', 'angEuler_1', 'angEuler_2', 'angEuler_3', 'FinOut_1', 'FinOut_2', 'FinOut_3', 'FinOut_4', 'err_FinBias_1', 'err_FinBias_2', 'err_FinBias_3', 'err_FinBias_4', 'FinCmd_1', 'FinCmd_2', 'FinCmd_3', 'FinCmd_4', 'err_BurnTime', 'err_Tilt_1', 'err_Tilt_2', 'err_delXcp']
                tr = tr[['Rmi_1', 'Rmi_2', 'Rmi_3', 'Vmi_1', 'Vmi_2', 'Vmi_3', 'Wmb_1', 'Wmb_2', 'Wmb_3', 'Accm_1', 'Accm_2', 'Accm_3', 'angEuler_1', 'angEuler_2', 'angEuler_3', 'FinOut_1', 'FinOut_2', 'FinOut_3', 'FinOut_4', 'FinCmd_1', 'FinCmd_2', 'FinCmd_3', 'FinCmd_4']].to_numpy()
                tr[0] = 4000 - tr[0]
                if self.model_type == 'A':
                    terminal = tr[-1].copy()
                    terminal[3]=terminal[4]=terminal[5]=terminal[6]=terminal[7]=terminal[8]=terminal[12]=terminal[13]=terminal[14]=terminal[19]=terminal[20]=terminal[21]=terminal[22]=0
                    tr = np.concatenate((tr, np.tile(terminal, (self.seq_len-len(tr), 1))), axis=0)
                    if i == 0:
                        data_x_raw = [tr for _ in range(500)]
                        data_y_raw = [i for _ in range(500)]
                    else:
                        data_x_raw.append(tr)
                        data_y_raw.append(i)
                else:
                    for idx in range(0, len(tr)-self.seq_len-self.pred_len+1, 10):
                        s_begin = idx
                        s_end = s_begin + self.seq_len
                        r_begin = s_end - self.label_len
                        r_end = r_begin + self.label_len + self.pred_len
                        
                        data_x_raw.append(tr[s_begin:s_end])
                        data_y_raw.append(tr[r_begin:r_end])
        
        self.data_x = np.array(data_x_raw)
        self.data_y = np.array(data_y_raw)

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]

        return x, y

    def __len__(self):
        return len(self.data_x)


class ForecastSet(Dataset):
    def __init__(self, path='Data/3D/', size=None, input='x_new.npy', output='y_new.npy', model_type='A', flag='train'):
        # size [seq_len, label_len, control_num]
        # info
        if size == None:
            if model_type == 'A':
                self.seq_len = 2656
                self.class_num = 5
            else:
                self.seq_len = 1992
                self.label_len = 664
                self.pred_len = 664
        else:
            if model_type == 'A':
                self.seq_len = size[0]
                self.class_num = size[1]
            else:
                self.seq_len = size[0]
                self.label_len = size[1]
                self.pred_len = size[2]
        self.model_type = model_type
        self.flag = flag
        
        # init
        self.path_x = path+input # 
        self.path_y = path+output # 
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
        data_x_raw = []
        data_y_raw = []
        tr = {}
        for i, tr_type in enumerate(['Normal', 'Burntime', 'Xcpposition', "ThrustTiltAngle", 'Finbias']):
            if i == 0:
                left = 0
                right = 1
            else:
                d_type = {'train':0, 'valid':1, 'test':2, 'pred':3}
                lefts = [0, int(500*0.7), int(500*0.8), 0]
                right = [int(500*0.7), int(500*0.8), 500, 500]
                left = lefts[d_type[self.flag]]
                right = right[d_type[self.flag]]
            for tr_i in range(left, right):
                tr = pd.read_csv(self.path_x + f'Type_{i+1}_{tr_i+1}.csv', header=None)
                tr.columns = ['Rmi_1', 'Rmi_2', 'Rmi_3', 'Vmi_1', 'Vmi_2', 'Vmi_3', 'Wmb_1', 'Wmb_2', 'Wmb_3', 'Accm_1', 'Accm_2', 'Accm_3', 'angEuler_1', 'angEuler_2', 'angEuler_3', 'FinOut_1', 'FinOut_2', 'FinOut_3', 'FinOut_4', 'err_FinBias_1', 'err_FinBias_2', 'err_FinBias_3', 'err_FinBias_4', 'FinCmd_1', 'FinCmd_2', 'FinCmd_3', 'FinCmd_4', 'err_BurnTime', 'err_Tilt_1', 'err_Tilt_2', 'err_delXcp']
                tr = tr[['Rmi_1', 'Rmi_2', 'Rmi_3', 'Vmi_1', 'Vmi_2', 'Vmi_3', 'Wmb_1', 'Wmb_2', 'Wmb_3', 'Accm_1', 'Accm_2', 'Accm_3', 'angEuler_1', 'angEuler_2', 'angEuler_3', 'FinOut_1', 'FinOut_2', 'FinOut_3', 'FinOut_4', 'FinCmd_1', 'FinCmd_2', 'FinCmd_3', 'FinCmd_4']].to_numpy()
                tr[0] = 4000 - tr[0]
                if self.model_type == 'A':
                    terminal = tr[-1].copy()
                    terminal[3]=terminal[4]=terminal[5]=terminal[6]=terminal[7]=terminal[8]=terminal[12]=terminal[13]=terminal[14]=terminal[19]=terminal[20]=terminal[21]=terminal[22]=0
                    tr = np.concatenate((tr, np.tile(terminal, (self.seq_len-len(tr), 1))), axis=0)
                    if i == 0:
                        data_x_raw = [tr for _ in range(500)]
                        data_y_raw = [i for _ in range(500)]
                    else:
                        data_x_raw.append(tr)
                        data_y_raw.append(i)
                else:
                    s_begin = 0
                    s_end = s_begin + self.seq_len
                    r_begin = 0
                    r_end = len(tr)
                    
                    data_x_raw.append(tr[s_begin:s_end])
                    data_y_raw.append(tr[r_begin:r_end])
        
        self.data_x = np.array(data_x_raw)
        self.data_y = np.array(data_y_raw)

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]

        return x, y

    def __len__(self):
        return len(self.data_x)
    

data_dict = {
    'train': ClassifySet,
    'test': ForecastSet,
}


def data_provider(args, flag):
    inpath = 'x_new.npy'
    outpath = 'y_new.npy'
    
    shuffle_flag = False
    drop_last = False
    batch_size = args.batch_size
    
    if args.model == 'Linear':
        size=[args.seq_len, args.label_len, args.control_num, args.pred_len]
    elif args.model == 'Transformer':
        size=[args.seq_len, args.label_len, args.control_num, args.step_len]
    
    path = args.root_path + args.data_path + f"{format(i, '03')}"
    
    data_set = ClassifySet(
        path=path,
        size=size,
        input = inpath,
        output = outpath, 
        model_type = args.model,
        flag = flag)
    
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