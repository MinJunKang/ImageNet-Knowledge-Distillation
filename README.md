# ImageNet-Knowledge-Distillation
Knowledge Distillation with ImageNet dataset

Paper : https://arxiv.org/abs/1503.02531

# Teacher Model
Resnet 18 (you can change with 50 or others)

# Student Model
MobileNet or DNN(only got < 20% test accuracy)

# Data
ImageNet Dataset : http://www.image-net.org/

# Training Method
We didn't used data augmentation technique(If there is time, try it!!)

Stop Training if overfitting(validation set accuracy doesn't increase) happens more than 50 epochs

# Programming Language
Python 3.6
Tensorflow, keras

# OS dependency
windows 10, ubuntu linux

# Result

<div class="imgTopic">
 <h1 class="title"><a href="#">Resnet - 18's Training result</a></h1>
 <p class="content"><a href="#"><img src="https://user-images.githubusercontent.com/29685163/49656613-20093680-fa81-11e8-97fb-ecb63c1ed385.png" alt="" width = "675" height ="350"/></a></p>
</div>

<div class="imgTopic2">
 <h1 class="title"><a href="#">MobileNet's Training result</a></h1>
 <p class="content"><a href="#"><img src="https://user-images.githubusercontent.com/29685163/49656474-cdc81580-fa80-11e8-8477-a5b31ab88e8c.png" alt="" width = "675" height ="350"/></a></p>
</div>

