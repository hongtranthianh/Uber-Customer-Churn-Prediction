import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score

def feature_important_ANOVA_f_test(X_train, y_train):
    '''
    Using ANOVA f_test to rank feature importance,
    the purpose is to choose features which are the most useful at predicting the target variable.
    '''
    #configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    #learn relationship from the training set
    fs.fit(X_train, y_train)
    #get sorted features and sorted importance scores
    features = X_train.columns
    importance = [fs.scores_[i] for i in range(len(fs.scores_))]
    sorted_index = np.argsort(-np.array(importance))
    features_sorted = [features[i] for i in sorted_index]
    importance_sorted = [importance[i] for i in sorted_index]
    return features_sorted, importance_sorted


def feature_importance_ANOVA_bar_chart(X_train, y_train):
    '''
    Plot bar chart for feature importance from ANOVA f_test
    '''
    features_sorted = feature_important_ANOVA_f_test(X_train, y_train)[0]
    importance_sorted = feature_important_ANOVA_f_test(X_train, y_train)[1]
    #plot the scores
    plt.barh(features_sorted, importance_sorted)
    plt.title('Feature importance (ANOVA f-test)');


def accuracy_by_top_k_features(cv, model, X_train, y_train):
    '''
    Bar chart showcases accuracy scores for each of top k importance features from ANOVA f_test
    Args:   
        cv: cross validation configuration (e.g., cv = KFold(n_splits=10, random_state=1, shuffle=True))
        model: classifier configuration (e.g., LogisticRegression(solver='liblinear'))
        X_train
        y_train
    '''
    # Get the list of sorted features by importance scores from ANOVA f-test
    features_sorted = feature_important_ANOVA_f_test(X_train, y_train)[0]
    # do cross validation for each top k of important features
    avg_accuracies = []
    num_feat = range(1, X_train.shape[1]+1)
    for k in num_feat:
        X_train_topk = X_train[X_train.columns.intersection(features_sorted[:k])]
        scores = cross_val_score(model, X_train_topk, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        avg_accuracy = np.mean(scores)*100
        avg_accuracies.append(avg_accuracy)
    #plot average accuracies
    plt.plot(num_feat, avg_accuracies)
    plt.locator_params(integer=True)
    plt.xlabel('k')
    plt.xticks(num_feat)
    plt.ylabel('Average accuracy')
    plt.title('Average accuracy by k most important features');

def get_k_with_highest_accuracy(cv, model, X_train, y_train):
    # Get the list of sorted features by importance scores from ANOVA f-test
    features_sorted = feature_important_ANOVA_f_test(X_train, y_train)[0]
    # do cross validation for each top k of important features
    max_accuracy = 0
    num_feat = range(1, X_train.shape[1]+1)
    for k in num_feat:
        X_train_topk = X_train[X_train.columns.intersection(features_sorted[:k])]
        scores = cross_val_score(model, X_train_topk, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        avg_accuracy = np.mean(scores)*100
        if avg_accuracy >= max_accuracy:
            max_accuracy = avg_accuracy
            max_acc_k = k
    print('At k = {}, we obtains the highest average accuracy = {}'.format(max_acc_k, max_accuracy))
