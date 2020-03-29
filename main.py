import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# adjusting the final representation to fit all plots into one single view
def adjust_n_show_plot():
    plt.subplots_adjust(left=0.09, right=0.94, wspace=0.5, hspace=0.52)
    plt.show()


# visualize by boxplot
def boxplot(col_name_list):
    for index in range(len(col_name_list)):
        plt.subplot(5, 6, index + 1)
        plt.boxplot(data[col_name_list[index]])
        plt.ylabel(col_name_list[index])
    adjust_n_show_plot()


# visualize by scatter graph
def scatterplot(col_name_list):
    for index in range(len(col_name_list)):
        plt.subplot(5, 6, index + 1)
        plt.scatter(data["diagnosis"], data[col_name_list[index]])
        plt.xlabel('diagnosis')
        plt.ylabel(col_name_list[index])
    adjust_n_show_plot()


# visualize by histogram
def histplot(col_name_list):
    for index in range(len(col_name_list)):
        plt.subplot(5, 6, index + 1)
        plt.hist(data[col_name_list[index]])
        plt.ylabel(col_name_list[index])
    adjust_n_show_plot()


# columns from 2 to end are features, column 2 is the target(class)
def get_separated_data(data_param):
    features_to_return = data_param.values[:, 1:-1]
    target_to_return = data_param.values[:, -1:]
    return features_to_return, target_to_return


# getting rid of that little rebels
def remove_outliers(data_param):
    cols_type_of_number = data_param.select_dtypes(include=["number"])
    cols_type_of_category = data_param.select_dtypes(exclude=["number"])
    non_outlier_element_ids = np.all(stats.zscore(cols_type_of_number) < 3, axis=1)
    clean_data = pd.concat([cols_type_of_number.loc[non_outlier_element_ids],
                            cols_type_of_category.loc[non_outlier_element_ids]], axis=1)
    return clean_data


# I tried to implement different types of classification algorithms to find the best accuracy so we have,
# Gaussian Naive Bayes method
def gaussian_naive_bayes(x_train, x_test, y_train):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train.ravel())  # here I used ravel() to flatten the 2D train_y array
    return gnb.predict(x_test)


# Logistic Regression method
def log_reg_model(x_train, x_test, y_train):
    lr = LogisticRegression(max_iter=7000)  # if the data is unnormalized, iteration might throw an error
    lr.fit(x_train, y_train.ravel())  # here I used ravel() to flatten the 2D train_y array
    return lr.predict(x_test)


# K Nearest Neighbor method
def k_nearest_neighbor(x_train, x_test, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train.ravel())  # here I used ravel() to flatten the 2D train_y array
    return knn.predict(x_test)


# Simple switch function for classification strategies
def classification_switch(argument, x_train, x_test, y_train):
    if argument is "gaussian_naive_bayes":
        return gaussian_naive_bayes(x_train, x_test, y_train)
    elif argument is "logistic_regression":
        return log_reg_model(x_train, x_test, y_train)
    elif argument is "k_nearest_neighbor":
        return k_nearest_neighbor(x_train, x_test, y_train)
    else:
        print("Invalid argument")
        return 0


def predict(data_param, is_preprocessed, classification_type):
    # cleaning the data
    clean_data = remove_outliers(data_param)
    # separating source and target columns
    features, target = get_separated_data(clean_data)
    # Pre-processing the data if the user enters true
    if is_preprocessed:
        features = MinMaxScaler().fit_transform(features)

    # setting the random state to make reproducible
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=.25, random_state=42)

    # implementing the entered classification strategy
    y_pred = classification_switch(classification_type, x_train, x_test, y_train)
    accuracy = accuracy_score(y_test, y_pred) * 100
    confusion_mx = confusion_matrix(y_test, y_pred)
    print("Classification method: " + classification_type +
          "\nIs it preprocessed: " + str(is_preprocessed) +
          "\nAccuracy Score: %" + str(accuracy) +
          "\nConfusion Matrix: \n" + str(confusion_mx))


def show_plots(col_names_param, flag):
    if flag:
        boxplot(col_names_param)  # good to see the amount of outliers
        scatterplot(col_names_param)  # good to see the correlation between target and features
        histplot(col_names_param)  # good to see the range where the features piling up


"""-------------------------- LETS START --------------------------"""

# read the data
data = pd.read_csv('Assignment/data/data.csv', sep=',')

# visualize each feature to the target variable to see the correlation, outliers etc
col_names = list(data.columns)
feature_col_names = col_names[2:]
show_plots(feature_col_names, flag=False)  # Switch the flag to True if you want to see the plots

start_time = time.time()  # time counter for the operation
# make the predict (classification types are:gaussian_naive_bayes,
#                                            logistic_regression,
#                                            k_nearest_neighbor)
predict(data, is_preprocessed=True, classification_type="k_nearest_neighbor")
print("Elapsed time:", time.time() - start_time, "seconds")
