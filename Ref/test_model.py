###
# import packages
###

import PIL
import torch
import tarfile
import sklearn
import torchvision

import numpy as np
import pandas as pd
import torch.nn as nn

from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score

###
# Set the parameters
###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 256 # 256 < 512
batch_size = 32
workers = 12

### Load the label data
test_df = pd.read_csv('./y_test.txt')

### Transform
test_trans = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.46888268 , 0.46888268 , 0.46888268], std=[0.30292818 , 0.30292818 , 0.30292818])
])

### dataset
class MIMICCXRDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, path_image, 
                 transform=transforms.Compose([transforms.ToTensor()])):
        self.dataframe = dataframe
        # Total number of datapoints
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform
        self.path_image = path_image
        
        self.PRED_LABEL = list(dataframe.iloc[:,  1:].columns)
        '''
            'Atelectasis',
            'Cardiomegaly',
            'Consolidation', 
            'Edema', 
            'Enlarged Cardiomediastinum', 
            'Fracture', 
            'Lung Lesion', 
            'Lung Opacity', 
            'No Finding', 
            'Pleural Effusion', 
            'Pleural Other', 
            'Pneumonia', 
            'Pneumothorax', 
            'Support Devices'
        '''
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        img = PIL.Image.open(self.path_image+item['study_id']+'.jpg')
        img = self.transform(img)

        # label = np.zeros(len(self.PRED_LABEL), dtype=int)
        label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
        for i in range(len(self.PRED_LABEL)):

            if (self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') == 1):
                label[i] = self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float')

        return img, label, item['study_id']#self.dataframe.`index`[idx]

### Get the test set.
tar = tarfile.open('./X_test.tar.gz', 'r:gz')
for tarinfo in tar:
    tar.extract(tarinfo, './')

test = MIMICCXRDataset(test_df, './X_test/', test_trans)
diseases = list(test_df.columns[1:])

### Get DataLoader
test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=True, num_workers=workers, pin_memory=True)

### test
model = torchvision.models.densenet121()
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(nn.Linear(num_ftrs, len(diseases)), nn.Sigmoid())
CheckPointData = torch.load('./model_parameter')
model.load_state_dict(CheckPointData['model'].module.state_dict())
model = model.to(device)

pred_df = pd.DataFrame(columns=['path'])
true_df = pd.DataFrame(columns=['path'])

for i, data in enumerate(test_loader):
    inputs, labels, study_id = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    true_labels = labels.cpu().data.numpy()

    batch_size = true_labels.shape

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

    # get predictions and true values for each item in batch
    for j in range(0, batch_size[0]):
        truerow = {}
        thisrow = {}

        truerow["path"] = study_id[j]
        thisrow["path"] = study_id[j]

        # iterate over each entry in prediction vector; each corresponds to
        # individual label
        for k in range(len(diseases)):
            thisrow["prob_" + diseases[k]] = probs[j, k]
            truerow[diseases[k]] = true_labels[j, k]
        pred_df = pred_df.append(thisrow, ignore_index=True)
        true_df = true_df.append(truerow, ignore_index=True)
    
target, predict = true_df.iloc[:, 1:].to_numpy().astype(int), pred_df.iloc[:, 1:].to_numpy()

try:
    macro_auroc = round(roc_auc_score(target, predict, average='macro'), 4)
    micro_auroc = round(roc_auc_score(target, predict, average='micro'), 4)
    macro_auprc = round(average_precision_score(target, predict, average='macro'), 4)
    micro_auprc = round(average_precision_score(target, predict, average='micro'), 4)
except BaseException:
    print("can't calculate auc for " + str(column))

with open('20214577_model.txt', 'w') as f:
    f.write(f'20214577\n{macro_auroc}\n{micro_auroc}\n{macro_auprc}\n{micro_auprc}\n')
print("Saved '20214577_model.txt'")