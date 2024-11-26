
import warnings
warnings.filterwarnings('ignore')

# data wrangling & pre-processing
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve


# model validation
from sklearn.metrics import log_loss ,roc_auc_score ,precision_score ,f1_score ,recall_score ,roc_curve ,auc
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score ,fbeta_score ,matthews_corrcoef
from sklearn import metrics

# cross validation
from sklearn.model_selection import StratifiedKFold

# machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier ,VotingClassifier ,AdaBoostClassifier ,GradientBoostingClassifier \
    ,RandomForestClassifier ,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC
# import xgboost as xgb

from scipy import stats

def printTitle(title):
    bold_title = f"\033[1m{title}\033[0m"
    print(f"{'-' *len(title)} \n{bold_title} \n{'-' *len(title)}")

def grab_col_names(df, cat_threshold=10, car_threshold=20):

    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]

    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_threshold and df[col].dtypes != "O"]

    cat_but_car = [col for col in df.columns if df[col].nunique() > car_threshold and df[col].dtypes == "O"]

    numeric_id = [col for col in df.columns if df[col].nunique() == df.shape[0] and max(df[col] ) -min(df[col]) == df.shape[0] - 1]

    cat_cols = sorted(list(set(cat_cols + num_but_cat) - set(cat_but_car)))

    num_cols = sorted([col for col in df.columns if df[col].dtypes != "O" and col not in num_but_cat])

    car_cols = cat_but_car + numeric_id

    return cat_cols, num_cols, car_cols


def DataFrame_Summary(df, title):
    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]} \n")

    printTitle(f"{title} DATAFRAME")

    printTitle(f"Head")
    display(df.head())

    printTitle(f"Tail")
    display(df.tail())

    groups = cat_cols, num_cols, car_cols = grab_col_names(df)

    for index, group in enumerate(groups):
        group_name = ['Categoric', 'Numeric', 'Cardinal'][index]
        printTitle(group_name + " Columns")

        if len(group) == 0:
            print(f"There is not any {group_name} Columns in the dataframe.\n")
        else:
            print(df[group].apply(lambda col: col.dtype), "\n")
            printTitle(f"Descriptive Statistics for {group_name} Columns")
            display(df[group].describe().T)
            print()


def num_col_summary(df, col, quantiles=[0, 0.05, 0.25, 0.5, 0.75, 0.95, 1], plot=False):
    display(pd.DataFrame(df[col].describe(quantiles).T))

    if plot:
        df[col].hist()
        plt.xlabel(col)
        plt.title(col)
        plt.show()


def cat_col_summary(df, col, plot=False):
    display(pd.DataFrame({col: df[col].value_counts(), "Ratio": 100 * df[col].value_counts() / len(df)}))

    if plot:
        sns.countplot(data=df, x=col, hue=col, palette='deep')
        plt.show()


def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    printTitle(numerical_col)
    df = dataframe.groupby(target).agg({numerical_col: "mean"})
    display(df)

    if plot:
        sns.barplot(x=df.index.tolist(), y=df.iloc[:,0], palette='deep')
        plt.show()


def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    printTitle(categorical_col)
    df = pd.DataFrame({"Disease Mean": dataframe.groupby(categorical_col)[target].mean(),
                       "Count": dataframe[categorical_col].value_counts(),
                       "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)})
    display(df)

    if plot:
        sns.barplot(x=df.index.tolist(), y=df.Count, palette='deep')
        plt.show()


def null_counts(df):
    printTitle("Null Counts")
    null_counts = df.isnull().sum()
    null_counts = pd.DataFrame(null_counts[null_counts != 0], columns=["Null Count"])
    null_counts["Ratio %"] = round(null_counts["Null Count"] * 100 / df.shape[0], 2)

    if null_counts.empty:
        print("There is not any Null Value in the dataframe.\n")
    else:
        null_counts = null_counts.sort_values(by="Ratio %", ascending=False)
        display(null_counts)
        print()

    return null_counts


def outlier_thresholds(df, col, q1=0.25, q3=0.75):
    quartile1 = df[col].quantile(q1)
    quartile3 = df[col].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit


def check_outlier(df, col, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(df, col, q1, q3)
    return True if df[(df[col] > up_limit) | (df[col] < low_limit)].any(axis=None) else False


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def BarChartForScores(scoreDF, score):
    fig = plt.figure(figsize=(8,5))
    sns.set_style('darkgrid')
    s = sns.barplot(x=scoreDF.index.tolist(), y=scoreDF[score], alpha=.8, palette=sns.color_palette())
    [s.bar_label(i, ) for i in s.containers]
    plt.xlabel("ML ALGORITHMS", fontsize=13)
    plt.ylabel(score, fontsize=15)
    plt.show()
    
def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 20))
    sns.set(font_scale=1)
    print()
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=5):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()



