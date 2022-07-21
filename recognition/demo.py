import string
import argparse

import torch
import torch.backends.cudnn as cudnn
# import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Dataset

from recognition.utils import CTCLabelConverter, AttnLabelConverter
from recognition.dataset import RawDataset, AlignCollate
from recognition.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

import os
from tqdm import tqdm
import json

path_root = './'

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default=os.path.join(path_root, 'debugs/crop_images'), help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default=os.path.join(path_root, 'models/TPS-ResNet-BiLSTM-Attn.pth'), help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    return opt

class ImgArrDataset(Dataset):
    def __init__(self, arr):
        # self.opt = opt
        self.arr = arr
        self.nSamples = len(self.arr)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        return (self.arr[index] , index)

class RECOGNITION():
    def __init__(self) -> None:
        self.opt = get_opt()
        self.model = None

        if 'CTC' in self.opt.Prediction:
            self.converter = CTCLabelConverter(self.opt.character)
        else:
            self.converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3

        self.AlignCollate_demo = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
    
    def get_model(self):
        if self.model is not None:
            return self.model

        self.model = Model(self.opt)
        self.model = torch.nn.DataParallel(self.model).to(device)

        # load model
        print('loading pretrained model from %s' % self.opt.saved_model)
        self.model.load_state_dict(torch.load(self.opt.saved_model, map_location=device))

        return self.model

    def predict_arr(self, bib_list):

        X = ImgArrDataset(bib_list)  
        data_loader = torch.utils.data.DataLoader(
            X,
            batch_size= self.opt.batch_size,
            shuffle=False,
            num_workers= int(self.opt.workers),
            collate_fn= self.AlignCollate_demo,
            pin_memory= True,
        )
        # predict
        model = self.get_model()
        model.eval()
        result_1_img = {}
        
        with torch.no_grad():
            for image_tensors, index in data_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in self.opt.Prediction:
                    preds = model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode(preds_index, preds_size)

                else:
                    preds = model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)                

                for index, (img_name, pred, pred_max_prob) in enumerate(zip(index, preds_str, preds_max_prob)):

                    if 'Attn' in self.opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    result_1_img[index] = (pred, round(float(confidence_score), 4))
  

        return result_1_img
