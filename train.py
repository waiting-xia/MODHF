import warnings, os
  
# os.environ["CUDA_VISIBLE_DEVICES"]="0"    

os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings('ignore')
from ultralytics import YOLO




if __name__ == '__main__':
  
    
    model = YOLO('/home/Code/yolo11-new/ultralytics/cfg/models/11/yolo11-hyper_DyHead.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    print(model.info())
    # model.train(data='/home/Code/yolo11-new/VOC.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=300,
    #             batch=32,
    #             close_mosaic=0, # 
    #             workers=4, # 
    #             device=1, # 
    #             optimizer='SGD', # using SGD
    #             # patience=0, # 
    #             # resume=True, # 
    #             # amp=False,
    #             # fraction=0.2,
    #             project='runs/train_hyper_dyhead_2',
    #             name='exp',
    #             )

