# Myers-Briggs Personality Type Classifier: Multiclass Classification using Machine Learning


The Myers–Briggs Type Indicator (MBTI) is a kind of psychological classification about human experience using four principal psychological functions, sensation, intuition, feeling, and thinking, constructed by Katharine Cook Briggs and her daughter Isabel Briggs Myers.

The Myers Briggs Type Indicator divides everyone into 16 distinct personality types across 4 axis:

Introversion (I) – Extroversion (E)
Intuition (N) – Sensing (S)
Thinking (T) – Feeling (F)
Judging (J) – Perceiving (P)

This project showcases the use of Natural Language Processing (NLP) with classification models from the [scikit-learn library](https://scikit-learn.org/stable/index.html) in identifying which personality type from the Myers-Briggs personality types classifier a given person is, based on their own written composition.

### Table of Contents

* [Problem Statement](#user-content-problem-statement)
* [Executive Summary](#user-content-executive-summary)
* [Data](#user-content-Data)
* [Modeling](#user-content-Modeling)
* [Conclusions and Recommendations](#user-content-conclusions-and-recommendations)

---

### Problem Statement

Can a machine learning model reliably predict a person's personality type?

---

### Executive Summary

The essential premise of this experiement was to use machine learning to create a distinction between the 16 different MBTI classifications. However, due to a stark contrast in class sizes I had to pivot the focus of the project to distinguish between the top 4 classes based on the amount of data we had on these classes. 

---

### Data

This data was collected through the PersonalityCafe forum, as it provides a large selection of people and their MBTI personality type, as well as what they have written. This dataset contains over 8600 rows of data, on each row is a person’s:

Type (This persons 4 letter MBTI code/type)
A section of each of the last 50 things they have posted (Each entry separated by "|||" (3 pipe characters))

#### Imbalance of Classes

There was a decision to make as far as how to handle the imbalanced classes that were represented in this dataset. To which I grappeled with the idea of imputing data into the lacking classes to have a more balanced datset, however, an issue with doing it this way is the inherent bias of the data augmenter(myself). What good a model would I be left with if much of the data was synthesized by hand? I ended up deciding to see if I can predict for the 4 most prominent personality types in the data set ('INFP', 'INFJ', 'INTP', 'INTJ'). Again, when dealing with messy, real-world data a practicing data scientist needs to make critical decisions about how to engineer the data that will affect the overall course of the project.  

The data in the repository are divided as such:

'clean_mbti_df.csv' contains data after proccessing and filtering out special characters, urls, and punctuations. This data is clean and ready for vectorization and modeling.

'heavy_sample.csv' contains data from the top four classes with the most amount of data. The only types in this set are the 'INFP', 'INFJ', 'INTP', 'INTJ' classes cleaned and ready for modeling.

'upper_sample.csv' contains data from the subset of data with the second-most amount of data. The only types in this set are the 'ENTP', 'ENFP', 'ISTP', 'ISFP' classes cleaned and ready for modeling.

'mbti_1.csv' contains the raw data as extracted from Kaggle

'pickle_model' contains the pickled model that was used to deploy on the Streamlit app

'pickle_vectorizer' contains the pickled vectorizer used to deploy on the Streamlit app

---

### Modeling

A sum total of 22 models were run to conduct this experiment utilizing sklearn's Logistic Regression, Multinomial NB, K Nearest Neighbors Classifier, and Random Forest. I also used the RidgeClassifierCV to perform regularization. Most of the models did poorly and overfit the data most likely due to the imbalanced classes that I chose not to impute for. Logistic Regression was the workhorse of the pack of models exhibiting the best scores. In addition, I performed ridge regularization to reduce the overfitting on the logistic regression models that did best and it did reduce the magnitude of overfitting.

I chose to vectorize the text data using the CountVectorizer due to prior experience and success using this tool. One can try on their own whichever vectorizer they prefer.

---

### Conclusions and Recommendations

In the end, can a machine learning model reliably predict a person's personality type? The answer is not relaibly, at least not this model.  

Overall, the biggest cripple to this experiment was the imbalance class types in the data which affected my models to classify better those classes with an abundance of datapoints. Despite using stratify equal to the 'y' variable, which is done during the train_test_split stage of modeling to ensure relatively equal proportions of each class are represented in the training and testing sets, it was still an issue that was very difficult to work around. The best score I was able to extrapolate from all the models used was from the RidgeClassifierCV model applied to the heavy_class subset of data. This yielded a training score of 82.7% and a testing score of 71.6%. 

There are various ways these scores can be attempted to be improved in future experiements. One method of further investigation involves imputing data into certain classes to equal that of the dominant class. In my experience, I have found success using the [KNN imputation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) technique. Another avenue of dealing with these imbalanced classes is to use [bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) techniques which utilize ensemble methods specific to working with a limited supply of data. Lastly, I suggest a future focus of this project to consider training the model on a single class to perform a binary classification experiment. Of course, working to represent all classes should be the long-term goal, but this will come as more data is collected.

