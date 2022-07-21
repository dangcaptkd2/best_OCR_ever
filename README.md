# best_OCR_ever
This repo release a module OCR for English and VietNamese

### Reference:
- Text detection: [TextFuseNet](https://github.com/ying09/TextFuseNet)
- Text English recognition: [DeepText](https://github.com/clovaai/deep-text-recognition-benchmark)
- Text Vietnamese recognition: [VietOCR](https://github.com/pbcquoc/vietocr)

### Setup:
###### Create and Activate conda enviroment
```
conda env create -f environment.yml
conda activate bestOCR
```
###### Download models
This project need 3 checkpoints: textfusenet, deeptext and vietocr. Create folder *models* and add checkpoints from [here](https://drive.google.com/drive/folders/1CvVelwRxO6CIBFTvGezXpOHRbycEQkMT?usp=sharing)

### Run
