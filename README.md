# COMS4045A NLP - Project: South African Fake news classifier

## About the project
The aim of this project is to detect whether South African based news articles are fake or legitimate.

## Implemented models
* Naive Bayes Classifier
* TextRNN: A bidirectional LSTM neural network used for text classification.

## Dataset

* The fake news are from [South African Disiniformation website data -2020](https://zenodo.org/record/4682843).
* The real news were scraped by me from [News24](https://www.news24.com/news24/southafrica).

I then merged the two datasets and shuffled them to create a dataset that I can use to train my classifier models.

## Getting started
* The Word embedding notebook was run on Google Colab using their GPU's.
* The Naive Bayes notebook can be run locally using the requirements below.

## Requirements
Recommended to use Anaconda for managing your environment.
1. Create a new environment using the `environment.yml` file:
    ```bash
        conda env create -f environment.yml
    ```
2. Activate the new environment
    ```bash
        conda activate ENV
    ```
3. Verify new environment was installed correctly
    ```bash
        conda env list
    ```

## Model Performance
| Metrics        | Naive Bayes          | textRNN  |
| ------------- |:-------------:| -----:|
| Valid Accuracy     | - | 0.54 |
| Test Accuracy      | 0.89      |   $0.48 |
| AUC| 0.95     |    0.56 |
| Precision| 0.89      |    0.24 |
| Recall| 0.89      |    0.50 |
| F1-score| 0.89      |    0.33 |

## References
* [Text Classification(tfidf vs word2vec vs bert)](https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794)
* [Text Analysis: Feature Engineering with NLP](https://towardsdatascience.com/text-analysis-feature-engineering-with-nlp-502d6ea9225d)
* [Fake news Detection using NLP techniques](https://medium.com/analytics-vidhya/fake-news-detection-using-nlp-techniques-c2dc4be05f99)
* [PyTorch Text Classification Tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#split-the-dataset-and-run-the-model)
* [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)
* [Lena Voita: NLP Course|For you](https://lena-voita.github.io/nlp_course.html)

## Future Work
* Improve Performance of the Recurrent Neural Network.
* Collect more data.
* Implement a [RCNN model](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552).
* Implement a Language Model such as [BERT](https://arxiv.org/abs/1810.04805).

