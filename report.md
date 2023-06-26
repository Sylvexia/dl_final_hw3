#  深度學習 作業三

## Abstract

This homework utilize SVM (Support Vector Machine), RandomForest, and custom Neural network to predict the bearing faults.

## Author 

Team Name: "Team FianCatto"

(Referring to a chess strategy: Fianchetto)

Member:

A1085125 洪祐鈞 

(Yes, there's only 1 person one the team and the professor is asking for a team name...)

## Few question to be answered at the first

Basically answering the professors requirement directly to save some time.

### What is the problem that you will be investigating? Why is it interesting?

Real life industrial problem is intriguing to me. Also it seems like its the most straight forward topic to investigate.

But actually I just choose the topic by elimination:

- I had a huge pain dealing with virtual try-on or image generation since my gpu memory is only 4 gigabyte, also kaggle would shut down your connection in 12 hours. I do not have good 

- I have no familiarity in stock market, if I were to do this experiment, single LSTM model would not just work. Also it might need to do a lot of web scraping in the process to get the data you need. It might be the most interesting topic since I might even do content analysis on stock news. However the topic is so big that a single person like me cannot handle.

- I have done pose estimation on jetson nano using ```Pose-ResNet18-Body``` when I was helping lab senior last year. I don't want to do the same task again. It was an interesting topic though.

- Heart disease prediction is quite similar to bearing fault classification. But I love doing task with multiple classification instead of binary classification.

### What are the challenges of this project?

Conversion of the dataset to feed to my own model took me some time, I was not familiar with data pre-processing. 
    
Also knowing why my model works is still a mystery to me.

### What dataset are you using? How do you plan to collect it?

CWRU dataset on [kaggle](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets), the csv dataset is relatively easy to process.

Not sure what does professor mean by "collect", maybe he was referring to the stock market exclusively, since the dataset of the task I did was relatively simple.

For the data preprocessing, I actually did standardize the CRWU dataset.

### What method or algorithm are you proposing? If there are existing implementations, will you use them and how? How do you plan to improve or modify such implementations?

Basically the task is extremely simple, the goal of the task is just extract 9 features and output 10 classification from 2300 data. Hence here we will compare the methods as following:

- Existing method:
    - Random Forest
    - Support Vector Machine
    - [FaultNet](https://arxiv.org/abs/2010.02146)

- Custom method:
    - Custom neural network that with only linear layers and relu activation layers.

Expectation:

I expect Neural Network approach would gain better result. And see if without 2D convolution layer proposed in the FaultNet would make more sense and make good prediction.

### What reading will you examine to provide context and background? If relevant, what papers do you refer to?

Reading: [FaultNet](https://arxiv.org/abs/2010.02146)

There's a interesting story behind it: 

When I saw FaultNet treat the data as a 2D data and utillze 2D convolution to classfy the bearing faults. It does not make any sense to me, since for what I've recall: 2D Convolution extract the neiborhood information with kernel sliding. 

The source code of the paper implementation basically sample 1600 data and shape to 40*40 data and do the convolution. It just sound extremely bizarre to me since the neiborhood data should be treated like it's independant to each other.

But their's accuracy is significant in the paper is better than me, so maybe I am wrong.

### How will you evaluate your results? Qualitatively, what kind of results do you expect (e.g. plots or figures)? Quantitatively, what kind of analysis will you use to evaluate and/or compare your results (e.g. what performance metrics or statistical tests)?

The evaluation and metrics will be shown later at the result section.

## Environment

- EndeavourOS (Cassini Nova) (Arch linux distro)
- Miniconda 23.1.1
- ASUS aspire-7 (A715-51G) laptop.
- CPU: intel i7-1260p
- GPU: NVIDIA GeForce RTX 3050 4GB Laptop GPU
- Python 3.11.3
- For python module version, please refer to requirements.txt

## How to Run?

```=bash
python svm.py
python random_forest.py
python neural network.py
```

3 python file represent 3 methods, after running the code, folder named with the method will generate result, plot and model checkpoint. Specifically:

```
svm
├── model.joblib
├── result.json
└── SVMconfusion_matrix.png

random_forest
├── feature_importance.png
├── model.joblib
├── RandomForestconfusion_matrix.png
└── result.json

neural_network
├── best_model.pth
├── loss_plot.png
├── NeuralNetworkconfusion_matrix.png
└── result.json
```
## Method and Result:

Basically all data is from ```feature_time_48k_2048_load_1.csv```, which can be download on [kaggle](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets)

The data is first read from csv file and interpret by pandas dataframe, then normalize 10 feature respecively.

### SVM

SVM (Support Vector Machine) is a machine learning technique that draws a line to separate different groups of data points. It helps classify things into different categories based on their features. It could also be used in non-linear classification.

The parameter basically is preset, I cannot get any better result with my own tuning.

### Random Forest

Random forest is a machine learning method that combines multiple decision trees to make predictions to extract the classification.

For all method, it perform the best. The parameter is still preset.

Also, since it's technically dicision tree, we can extract the importance of features respectively:

### Neural Network

Since the task itself is just a classifcation task, we can build a neural network to do the classification task:

The model architecture:

```
NN(
  (fc): Sequential(
    (0): Linear(in_features=9, out_features=36, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=36, out_features=144, bias=True)
    (3): ReLU(inplace=True)
    (4): Linear(in_features=144, out_features=288, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=288, out_features=64, bias=True)
    (7): ReLU(inplace=True)
    (8): Linear(in_features=64, out_features=32, bias=True)
    (9): ReLU(inplace=True)
    (10): Linear(in_features=32, out_features=16, bias=True)
    (11): ReLU(inplace=True)
    (12): Linear(in_features=16, out_features=10, bias=True)
  )
)
```

## Conclusion

- 2300 data is not a big number, basically it can be done with simple machine learning technique without deep learning. Which still get good result in small amount of runtime.