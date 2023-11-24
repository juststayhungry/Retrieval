##Getting started

###Requirements

```
cd Retrieval
pip install -r requirements.txt
```

###download dataset
#####download coco caption dataset

```
mkdir data/mscoco
wget http://images.cocodataset.org/zips/train2017.zip  -O data/mscoco
unzip /dataset/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/mscoco
unzip /dataset/annotations_trainval2017.zip
```

#####download czsl's dataset
follow https://github.com/BatsResearch/csp/blob/main/download_data.sh to install the czsl's datasets.

###Training
To train a model, the command is simply:

```
python train.py --train-type --dataset --batch-size
```

###Evaling
To Evaluate a model, the command is simply:

```
python eval.py --pretrained-pt # path/to/trained model's checkpoints/epoch_K.pt--dataset --batch-size --openai_clip_model# the openai's clip model for eval[choices from "RN101","RN50","ViT-B/32"]
```

##Acknowledgment

Our codebase builds upon several existing publicly available codes. Specifically, we have modified and integrated the following repos into this project:

* https://github.com/openai/CLIP
* https://github.com/mlfoundations/open_clip
