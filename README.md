# Uber-Customer-Churn-Prediction

<br>
<div align="center">
<img src="https://cdn.britannica.com/72/239572-050-F878B4FD/Uber-driver-holds-smartphone-in-car.jpg">
</div>
<br>

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [How to interact](#interact)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## 1. Installation <a name="installation"></a>
- No need to install libraries beyond the Anaconda distribution of Python.
- The code should run with no issues using Python versions 3.*. Currently using `Python 3.11.3` on `Windowns 11`.

## 2. Project Motivation<a name="motivation"></a>
**Ride-hailing** is an on-demand transportation service that provides an efficient travel mode by matching drivers and travelers via smartphone apps. This industry is very dynamic and gives its customers lots of choices. However, customers become more knowledgeable and less patient nowadays, which easily leads them to switch to competitors (aka **customer churn**)

To meet the need of surviving in this competitive industry, the retention of existing customers becomes a huge challenge. Because retaining an existing customer is a much lower cost than acquiring a new customer, to have a better customer retention strategy, this project will use transactional data about Uber trips as an experiment doing the following:
* Identify trends in riders’ activity
* Build a prediction model with **Logistic Regression** to detects who is likely to churn in the next 3 months.
* Segment churners.

## 3. File Descriptions <a name="files"></a>

```

- classifier
|- Build classifier.ipynb  # modeling process from scratch
|- classifier.pkl  # model saved from "Build classifier.ipynb"
                   # Whenever we need to use this model to make prediction,
                   # just load this model (pickled_model = pickle.load(open('model.pkl', 'rb')))
|- functions.py # all functions for do cross validation in "Build classifier.ipynb"

- data
|- feature.csv # saved from "Feature engineering.ipynb", used for modeling in "Build classifier.ipynb"
|- prediction.csv # prediction result saved from "Build classifier.ipynb"
|- uber_peru_2010.csv # original data to used in this repo

- Images # a folder contains images used in README

- Churn segmentation.ipynb # Segment churners

- EDA.ipynb # Exploratory Data Analysis

- Feature engineering.ipynb # to feed data for classifier

- README.md

```

## 4. How to interact <a name="interact"></a>

**Step 1**: Pull this repository to your local machine
```
git clone https://github.com/hongtranthianh/Uber-Customer-Churn-Prediction.git
```

**Step 2**: Run `functions.py` file
```
cd classifier
```

```
py functions.py
```

**Step 2**: Find below notebooks for specific purpose
* Get to know the data and do EDA: [EDA.ipynb](https://github.com/hongtranthianh/Uber-Customer-Churn-Prediction/blob/main/EDA.ipynb)
* Build a prediction model with **Logistic Regression** to detects who is likely to churn in the next 3 months:
    - [Feature engineering.ipynb](https://github.com/hongtranthianh/Uber-Customer-Churn-Prediction/blob/main/Feature%20engineering.ipynb)
    - [Build classifier.ipynb](https://github.com/hongtranthianh/Uber-Customer-Churn-Prediction/blob/main/classifier/Build%20classifier.ipynb)
* Segment churners: [Churn segmentation.ipynb](https://github.com/hongtranthianh/Uber-Customer-Churn-Prediction/blob/main/Churn%20segmentation.ipynb)

## 5. Results<a name="results"></a>
The main findings of the code can be found at the post available [here](https://medium.com/@ttah.hongtran/customer-churn-prediction-and-analysis-for-uber-ride-hailing-service-fbf1dc75a848)

## 6. Licensing, Authors, Acknowledgements<a name="licensing"></a>

Data is directly taken from [Kaggle](https://www.kaggle.com/datasets/marcusrb/uber-peru-dataset) with license: Data files © Original Authors

Any copies from this repo must be under 30%. If over 30%, you have to attach a direct link to this repo in your work.
