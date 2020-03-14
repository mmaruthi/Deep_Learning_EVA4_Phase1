
Assignment 8 - EVA 4 :

Solution acheived and the approach followed :

1. Used ResNet 18 model from the Repo and added it to our modular code repo.

2. Changed the ResNet18 model a bit - Removed Fully connected Layer , Added Softmax for determining the output, Used a convolutionlayer after GAP.

[ResNet Model](https://github.com/mmaruthi/Deep_Learning_EVA4_Phase1/blob/master/S8_ResNet_Modular/s8_resnet.py)

3. Used the rest of the things - Transforms , Train Loader , Loss calculation from the [modular code](https://github.com/mmaruthi/Deep_Learning_EVA4_Phase1/tree/master/S8_ResNet_Modular/model)

4. Acheived a target of 85% . Highest accuracy got is 87.5% 

5. Trying from my side to tune the model further to reduce overfitting 
