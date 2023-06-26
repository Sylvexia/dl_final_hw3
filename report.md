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

Basically answering the professors requirement directly.

### What is the problem that you will be investigating? Why is it interesting?

Real life industrial problem is intriguing to me. Also it seems like its the most straight forward topic to investigate.

But actually I just choose the topic by elimination:

- I had a huge pain dealing with virtual try-on or image generation since my gpu memory is only 4 gigabyte, also kaggle would shut down your connection in 12 hours. I do not have good 

- I have no familiarity in stock market, if I were to do this experiment, single LSTM model would not just work. Also it might need to do a lot of web scraping in the process to get the data you need. It might be the most interesting topic since I might even do content analysis on stock news. However the topic is so big that a single person like me cannot handle.

- I have done pose estimation on jetson nano using ```Pose-ResNet18-Body``` when I was helping lab senior. I don't want to do the same task again. It was an interesting topic

- Heart disease prediction is quite similar to bearing fault classification. But I love doing task with multiple classification instead of binary classification.

### What are the challenges of this project?

Conversion of the dataset to feed to my own model took me some time, I was not familiar with data pre-processing. 
    
Also knowing why my model works is still a mystery to me.

### What dataset are you using? How do you plan to collect it?

CWRU dataset on [kaggle](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets), the csv dataset is relatively easy to process.

### What method or algorithm are you proposing? If there are existing implementations, will you use them and how? How do you plan to improve or modify such implementations?

- Existing method:
    - Random Forest
    - Support Vector Machine
    - [FaultNet](https://arxiv.org/abs/2010.02146)

- Custom method:
    - Custom neural network that with only linear layers and relu activation layers.



### What reading will you examine to provide context and background? If relevant, what papers do you refer to?

Reading: [FaultNet](https://arxiv.org/abs/2010.02146)

There's a interesting story behind it: 

When I saw FaultNet treat the data as a 2D data and utillze 2D convolution to classfy the bearing faults. It's just does not make any sense to me, since for what I've recall: 2D Convolution extract the neiborhood information with kernel sliding. 

The source code of the paper implementation basically sample 1600 data and shape to 40*40 data and do the convolution. It just sound extremely bizarre to me since the neiborhood data should be treated like it's independant to each other.

But their's accuracy is significant better than me, so maybe I am wrong.

### How will you evaluate your results? Qualitatively, what kind of results do you expect (e.g. plots or figures)? Quantitatively, what kind of analysis will you use to evaluate and/or compare your results (e.g. what performance metrics or statistical tests)?

-