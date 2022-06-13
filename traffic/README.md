# ai50-project5-traffic

## Specification: https://cs50.harvard.edu/ai/2020/projects/5/traffic/

## Summary (as required in the project specification):
In order to determine the optimal parameters for the convolutional neural network model for this project, I started with the values applied in handwriting.py (from the lecture) and measured the average training accuracy, test accuracy and processing time per step (epoch) over 3 trials. Next, I experimented with changing the paremeters for the convolutional layer, pooling layer, hidden layer and dropout in isolation (i.e. when testing parameters for convolutional layer, all other parameters remained as baseline values), and then determined the parameter values that resulted in the relative best test accuracy in consideration of the processing time involved.

A summary of these results is presented below (details are in the section "Documenting test results for changing the neural network model").

1. Convolutional layer = Increasing the number of filters beyond 32 increased the processing time with no significant increase in accuracy, while fewer filters resulted in lower accuracy. Further, a filter size of 5x5 provided better results than 3x3, improving accuracy for a minor increase in procesing time. Larger filter sizes did not significantly improve accuracy despite increased processing time.
2. Max pooling layer = 2x2 is the best parameter for MaxPool for this model. Increasing the pool size reduced the accuracy of the model although there was lower processing time, while reducing to 1x1 greatly increased the processing time for the model and also reduced accuracy.
3. Number of convolutional and pooling layers = Adding another set of Conv2D and MaxPool layers improved accuracy, but adding a 3rd set of layers resulted in lower accuracy than 2 layers.
4. Hidden layer = Among the test cases, using more than 256 units for the hidden layer did not result in a significant increase in test accuracy despite an increase in processing time. 3 hidden layers significantly improved test accuracy vs 1 hidden layer with only a minimal increase in processing time. Using more than 3 layers did not improve test accuracy.
5. Dropout value = Higher dropout value = lower training accuracy. No significant changes in processing time when changing dropout value. In terms of test accuracy, we get the best results at 0.3 dropout value, although the difference with 0.25 dropout value was not significant.

Next, I tested the combined best parameters above and discovered that although the parameters described above were the best when changed in isolation, combining them altogether did not result in the best accuracy. Thus, I experimented with different combinations of the above best parameters to get the best test accuracy (details in section "Documenting test results for using a combination of the best parameter values from the previous section:"), and concluded with the final model of **2 layers of Conv2D 3x3, 32 units and MaxPool 2x2, followed by a hidden layer of 256 units and lastly a dropout value of 0.5**.

## Final model results (after running tests described in the next section below) ##
(Conv2D 32, 3x3; MaxPool 2x2) x 2; hidden layer 256; dropout 0.5:

    i. Training accuracy: 0.9730, 0.9736, 0.9735 (0.9734)

    ii. Testing accuracy: 0.9858, 0.9886, 0.9863 (0.9869)

    iii. Time: 42-45s/step, 38-41s/step, 41-42/step


## Documenting test results for changing the neural network model:

### Starting values - using handwriting.py from the lecture as baseline (runnning tests 3 times):
Conv2D 32, 3x3; MaxPool 2x2; hidden layer 128; dropout 0.5 (Using values in handwriting.py as starting values):
    
    i. Training accuracy: 0.8611, 0.8914, 0.8686 (0.8737)
    
    ii. Testing accuracy: 0.9599, 0.9687, 0.9677 (0.9654)
    
    iii. Time: 31-34s/step, 31-39s/step, 30-41s/step

### Changing # of filters in the Conv2D layer (all other values same as baseline):
1. Conv2D 16:
    
    i. Training accuracy: 0.7832
    
    ii. Testing accuracy: 0.9489
    
    iii. Time: 23-25s/step

2. Conv2D 24:
    
    i. Training accuracy: 0.8143
    
    ii. Testing accuracy: 0.9520
    
    iii. Time: 29-30s/step

3. **Conv2D 32** (status quo)

4. Conv2D 48:
    
    i. Training accuracy: 0.8916, 0.8516, 0.8510 (0.8647)
    
    ii. Testing accuracy: 0.9723, 0.9670, 0.9622 (0.9672)
    
    iii. Time: 40-41s/step, 38-40s/step, 39-43s/step

5. Conv2D 64:
    
    i. Training accuracy: 0.8934, 0.9216, 0.8681 (0.8943)
    
    ii. Testing accuracy: 0.9665, 0.9737, 0.9726 (0.9709)
    
    iii. Time: 54-61s/step, 51-53s/step, 52s/step

6. Conv2D 96:

    i. Training accuracy: 0.9179, 0.9067, 0.8485 (0.8910)

    ii. Testing accuracy: 0.9735, 0.9730, 0.9642 (0.9709)

    iii. Time: 71-84s/step, 70-71s/step, 68-70s/step

7. Conv2D 128:
    
    i. Training accuracy: 0.8493, 0.8666, 0.9082 (0.8747)
    
    ii. Testing accuracy: 0.9613, 0.9679, 0.9752 (0.9681)
    
    iii. Time: 88-94s/step, 102-122s/step, 99-113s/step

8. Conv2D 256:

    i. Training accuracy: 0.8959, 0.8515

    ii. Testing accuracy: 0.9712, 0.9668

    iii. Time: 180-181s/step, 184-213s/step

**Takeaway:** Increasing the number of filters beyond 32 increased the processing time with no significant increase in accuracy, while fewer filters resulted in lower accuracy.

### Changing sizes of filters in the Conv2D layer (all other values same as baseline):

**1. Conv2D 5x5:** (+0.0078 accuracy, +10s/step)

    i. Training accuracy: 0.9187, 0.9132, 0.9011 (0.911)

    ii. Testing accuracy: 0.9781, 0.9717, 0.9697 (0.9732)

    iii. Time: 40-44s/step, 40-41s/step, 41-50s/step

2. Conv2D 7x7:

    i. Training accuracy: 0.9137, 0.9133, 0.8938 (0.9069)

    ii. Testing accuracy: 0.9727, 0.9674, 0.9716 (0.9706)

    iii. Time: 52-55s/step, 54-59s/step, 52-55s/step

3. Conv2D 9x9:

    i. Training accuracy: 0.8673, 0.9094, 0.8960 (0.8909)
    
    ii. Testing accuracy: 0.9617, 0.9714, 0.9712 (0.9681)
    
    iii. Time: 50-53s/step, 48-49s/step, 51-55s/step

**Takeaway:** 5x5 is the best parameter for Conv2D for this model, improving accuracy at the expense of a minor increase in procesing time. Larger sizes did not significantly improve accuracy despite increased processing time.

### Changing pool size for MaxPool layer:
1. MaxPool 1x1:

    i. Training accuracy: 0.8342
    
    ii. Testing accuracy: 0.9562
    
    iii. Time: 73-77s/step

**2. MaxPool 2x2:** (status quo)

3. MaxPool 3x3:

    i. Training accuracy: 0.8435, 0.8560, 0.8599 (0.8531)
    
    ii. Testing accuracy: 0.9451, 0.9593, 0.9520 (0.9521)
    
    iii. Time: 25-26s/step, 24-27s/step, 25-29s/step

4. MaxPool 5x5:

    i. Training accuracy: 0.8330, 0.8508, 0.8762 (0.8532)
    
    ii. Testing accuracy: 0.9471, 0.9505, 0.9575 (0.9517)
    
    iii. Time: 23-25s/step, 24-26s/step, 23-25s/step

**Takeaway:** 2x2 remains as the best parameter for MaxPool for this model. Increasing the pool size reduced the accuracy of the model although there was lower processing time, while reducing to 1x1 greatly increased the processing time for the model and also reduced accuracy

### Increasing # of convolutional and pooling layers
**1. (Conv2D; MaxPool) X 2:** (+0.0171 accuracy, +6s/step)

    i. Training accuracy: 0.9510, 0.9361, 0.9444 (0.9438)
    
    ii. Testing accuracy: 0.9860, 0.9786, 0.9828 (0.9825)
    
    iii. Time: 37-53s/step, 37-40s/step, 37-50s/step

2. (Conv2D 32; MaxPool) X 3:

    i. Training accuracy: 0.9413, 0.9507, 0.9484 (0.9468)
    
    ii. Testing accuracy: 0.9690, 0.9733, 0.9718 (0.9713)
    
    iii. Time: 41-50s/step, 38-39s/step, 36-37s/step

**Takeaway:** Adding another set of Conv2D and MaxPool layers improved accuracy, but adding a 3rd set of layers resulted in lower accuracy than 2 layers.

### Changing size of hidden layer:
1. hidden layer 32:

    i. Training accuracy: 0.3382
    
    ii. Testing accuracy: 0.6956
    
    iii. Time: 22-25s/step

2. hidden layer 64:

    i. Training accuracy: 0.8119, 0.8092, 0.6508 (0.7573)
    
    ii. Testing accuracy: 0.9499, 0.9574, 0.9108 (0.9394)
    
    iii. Time: 25-29s/step, 28-30s/step, 27-31s/step

**3. hidden layer 256:** (+0.0104 accuracy, +20s/step)

    i. Training accuracy: 0.9517, 0.9525, 0.9515 (0.9519)
    
    ii. Testing accuracy: 0.9751, 0.9768, 0.9736 (0.9758)
    
    iii. Time: 45-59s/step, 47-52s/step, 51-60s/step

4. hidden layer 512:

    i. Training accuracy: 0.9743, 0.9701, 0.9690 (0.9711)
    
    ii. Testing accuracy: 0.9756, 0.9800, 0.9791 (0.9782)
    
    iii. Time: 53-54s/step, 59-67s/step, 60-66s/step

5. hidden layer 640:

    i. Training accuracy: 0.9730, 0.9697, 0.9730 (0.9719)
    
    ii. Testing accuracy: 0.9809, 0.9795, 0.9804 (0.9803)
    
    iii. Time: 56-60s/step, 56-61s/step, 60-62s/step

5. hidden layer 800:

    i. Training accuracy: 0.9754, 0.9774, 0.9728 (0.9752)
    
    ii. Testing accuracy: 0.9808, 0.9758, 0.9806 (0.9790)
    
    iii. Time: 66-67s/step, 62-65s/step, 72-82s/step

**Takeaway:** Among the test cases, using more than 256 units for the hidden layer did not result in a significant increase in test accuracy despite an increase in processing time.

### Increasing # of hidden layers
1. hidden layer x 2:

    i. Training accuracy: 0.9751, 0.9718, 0.9728 (0.9732)
    
    ii. Testing accuracy: 0.9655, 0.9679, 0.9716 (0.9683)
    
    iii. Time: 32-35s/step, 30-36s/step, 31-32s/step

**2. hidden layer x 3:** (+0.0060 accuracy, -2s/step)

    i. Training accuracy: 0.9729, 0.9777, 0.9724 (0.9743)
    
    ii. Testing accuracy: 0.9689, 0.9731, 0.9723 (0.9714)
    
    iii. Time: 31-34s/step, 31-36s/step, 32-37s/step

3. hidden layer x 4:

    i. Training accuracy: 0.9762, 0.9748, 0.9774 (0.9761)
    
    ii. Testing accuracy: 0.9550, 0.9566, 0.9570 (0.9562)
    
    iii. Time: 31-35s/step, 36-40s/step, 34-38s/step

4. hidden layer x 5:

    i. Training accuracy: 0.9715, 0.9639, 0.9714 (0.9689)
    
    ii. Testing accuracy: 0.9567, 0.9486, 0.9558 (0.9537)
    
    iii. Time: 37-50s/step, 36-42s/step, 34-43s/step

**Takeaway:** 3 hidden layers significantly improved test accuracy vs 1 hidden layer with only a minimal increase in processing time. Using more than 3 layers did not improve test accuracy.

### Changing dropout value:
1. Dropout 0.1:

    i. Training accuracy: 0.9799, 0.9808, 0.9765 (0.9791)
    
    ii. Testing accuracy: 0.9718, 0.9601, 0.9675 (0.9665)
    
    iii. Time: 32-33s/step, 34-38s/step, 31-33s/step

2. Dropout 0.2:

    i. Training accuracy: 0.9660, 0.9680, 0.9632 (0.9657)
    
    ii. Testing accuracy: 0.9728, 0.9631, 0.9713 (0.9691)
    
    iii. Time: 32-39s/step, 33-41s/step, 31-34s/step

3. Dropout 0.25:

    i. Training accuracy: 0.9530, 0.9700, 0.9523 (0.9584)
    
    ii. Testing accuracy: 0.9660, 0.9805, 0.9731 (0.9732)
    
    iii. Time: 31-40s/step, 32-33s/step, 37-39s/step

**4. Dropout 0.3:** (+0.0088 accuracy, +3s/step)

    i. Training accuracy: 0.9404, 0.9542, 0.9558 (0.9501)
    
    ii. Testing accuracy: 0.9722, 0.9737, 0.9718 (0.9742)
    
    iii. Time: 35-39s/step, 35-48s/step, 35-42s/step

5. Dropout 0.4:

    i. Training accuracy: 0.9474, 0.9020, 0.9399 (0.9298)
    
    ii. Testing accuracy: 0.9726, 0.9576, 0.9711 (0.9671)
    
    iii. Time: 33-37s/step, 36-42s/step, 31-35s/step

6. Dropout 0.6:

    i. Training accuracy: 0.7375, 0.8337, 0.8722 (0.8145)
    
    ii. Testing accuracy: 0.9469, 0.9599, 0.9643 (0.9570)
    
    iii. Time: 34-50s/step, 33-36s/step, 31-39s/step

7. Dropout 0.75:

    i. Training accuracy: 0.5723, 0.5140 
    
    ii. Testing accuracy: 0.9108, 0.8769
    
    iii. Time: 31-32s/step, 31-32s/step

8. Dropout 0.9:

    i. Training accuracy: 0.2513
    
    ii. Testing accuracy: 0.6475
    
    iii. Time: 30-31s/step

**Takeaway:** Higher dropout value = lower training accuracy. No significant changes in processing time when changing dropout value. In terms of test accuracy, we get the best results at 0.3 dropout value, although the difference with 0.25 dropout value was not significant.

### Summary of best results for each parameter:

**Conv2D 32** (status quo) (0.9654)

**Conv2D 5x5:** (+0.0078 accuracy, +10s/step) (0.9732)

**MaxPool 2x2:** (status quo) (0.9654)

**(Conv2D; MaxPool) X 2:** (+0.0171 accuracy, +6s/step) (0.9825)

**Hidden layer 256:** (+0.0104 accuracy, +20s/step) (0.9758)

**Hidden layer x 3:** (+0.0060 accuracy, -2s/step) (0.9714)

**Dropout 0.3:** (+0.0088 accuracy, +3s/step) (0.9742)

## Documenting test results for using a combination of the best parameter values from the previous section:

Next I experiment different combinations of the above best parameters to get the best test accuracy:
1. (Conv2D 32, 5x5; MaxPool 2x2) x 2; hidden layer 256 x 3; dropout 0.3:

    i. Training accuracy: 0.9882, 0.9899, 0.9886 (0.9889)
    
    ii. Testing accuracy: 0.9810, 0.9731, 0.9672 (0.9737)
    
    iii. Time: 68-81s/step, 66-79s/step, 60-67s/step

2. (Conv2D 32, 3x3; MaxPool 2x2) x 2; hidden layer 256 x 3; dropout 0.3:

    i. Training accuracy: 0.9882, 0.9858, 0.9891 (0.9877)
    
    ii. Testing accuracy: 0.9656, 0.9809, 0.9794 (0.9753)
    
    iii. Time: 42-53s/step, 46-52s/step, 55-83s/step

3. (Conv2D 32, 3x3; MaxPool 2x2) x 2; hidden layer 128 x 3; dropout 0.3:

    i. Training accuracy: 0.9824, 0.9812, 0.9840 (0.9825)
    
    ii. Testing accuracy: 0.9750, 0.9768, 0.9786 (0.9768)
    
    iii. Time: 36-39s/step, 37-44s/step, 37-41s/step

4. (Conv2D 32, 3x3; MaxPool 2x2) x 2; hidden layer 256; dropout 0.3:

    i. Training accuracy: 0.9838, 0.9815, 0.9837 (0.9830)
    
    ii. Testing accuracy: 0.9837, 0.9854, 0.9854 (0.9848)
    
    iii. Time: 42-44s/step, 43-49s/step, 41-44s/step

5. (Conv2D 32, 5x5; MaxPool 2x2) x 2; hidden layer 256; dropout 0.3:

    i. Training accuracy: 0.9882, 0.9875, 0.9861 (0.9873)
    
    ii. Testing accuracy: 0.9862, 0.9852, 0.9812 (0.9842)
    
    iii. Time: 54-56s/step, 55-58s/step, 60-67s/step

7. (Conv2D 32, 3x3; MaxPool 2x2) x 2; hidden layer 256; dropout 0.5:

    i. Training accuracy: 0.9730, 0.9736, 0.9735 (0.9734)
    
    ii. Testing accuracy: 0.9858, 0.9886, 0.9863 (0.9869)
    
    iii. Time: 42-45s/step, 38-41s/step, 41-42/step
