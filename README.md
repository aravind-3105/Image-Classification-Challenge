# Image Classification
Chunin Exams Food Track - Food Classification Challenge

The CSV file generated achieves a F1-score of 0.479.

**Challenge Link:** https://www.aicrowd.com/challenges/chunin-exams-food-track-cv-2021)

**Problem Statement:**

Maintaining a healthy diet is difficult. As the saying goes, the best way to escape a problem is to solve it. So why not leverage the power  of deep learning and computer vision to build the foundation of a  semi-automated food tracking application?

With over 9300 hand-annotated images with 61 classes, the challenge  is to train accurate models that can look at images of food items and  detect the food items present in the image. It's time to unleash the  food (data)scientist in you! Given any `image`, `identify` the `food` item present in it.

**Install dependencies:**

`pip3 install -r requirements.txt`

**Obtain Dataset:**

```python
!wget -q https://datasets.aicrowd.com/default/aicrowd-practice-challenges/public/foodc/v0.1/train_images.zip
!wget -q https://datasets.aicrowd.com/default/aicrowd-practice-challenges/public/foodc/v0.1/test_images.zip
!wget -q https://datasets.aicrowd.com/default/aicrowd-practice-challenges/public/foodc/v0.1/train.csv
!wget -q https://datasets.aicrowd.com/default/aicrowd-practice-challenges/public/foodc/v0.1/test.csv
```

```python
!mkdir data
!mkdir data/test
!mkdir data/train
!unzip train_images -d data/train
!unzip test_images -d data/test
```

Analysis is done by:

1. Adding Batch Norm
2. Adding new Layers
3. With Dropout
4. Different activation functions at the end
5. Different Optimizers
6. Data Augmentation like color jitter, rotation

**Final model:**

resnet18 is used for the submission. resnet50,resnet101 can be also be used for the same purpose
