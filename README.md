##Getting started

###Requirements

```
cd Retrieval
pip install -r requirements.txt
```

###download dataset

```
mkdir data/mscoco
wget http://images.cocodataset.org/zips/train2017.zip  -O data/mscoco
unzip /dataset/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/mscoco
unzip /dataset/annotations_trainval2017.zip
```

###Training
To train a model, the command is simply:

```
python train.py --img_dir "path of directory containing training images" --train_type 
```

##Acknowledgment

Our codebase builds upon several existing publicly available codes. Specifically, we have modified and integrated the following repos into this project:

* https://github.com/openai/CLIP
* https://github.com/mlfoundations/open_clip
