from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import os
from PIL import Image

# sys.path.append()

class RECOGNITION_VN():
    def __init__(self) -> None:
        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['weights'] = './models/transformerocr.pth'
        self.config['cnn']['pretrained']=False
        self.config['device'] = 'cuda'
        self.config['predictor']['beamsearch']=False

        self.model = None

    def get_model(self):
        if self.model is not None:
            return self.model         
        self.model = Predictor(self.config)
        return self.model


    def predict(self, image_array):
        model = self.get_model()
        text_boundings = [Image.fromarray(B) for B in image_array]
        texts, scores = model.predict_batch(text_boundings,return_prob=True)
        print(">>>>>>>",dict(zip(texts, scores)))
        final_text = [texts[i] for i in range(len(texts)) if scores[i]>0.6]
        return final_text
    
