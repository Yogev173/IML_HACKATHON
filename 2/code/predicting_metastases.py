####### IMPORTS #######
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from predicting_metastases_filter import filter_data
#######################

def load_data(file_path):
        """
        Loads data from a CSV file and returns a pandas DataFrame.
        """
        data = pd.read_csv(file_path)
        return data


def split_data(data, target_columns, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.
        Returns X_train, X_test, y_train, y_test.
        """
        X = data.drop(target_columns, axis=1)
        y = data[target_columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test


# Note: this function is for automatically selecting features, may be unnecessary for out project
def filter_features(X_train, X_test, y_train, k=10):
        """
        Applies feature selection using chi-square test and selects top k features.
        Returns X_train_filtered, X_test_filtered.
        """
        selector = SelectKBest(score_func=chi2, k=k)
        X_train_filtered = selector.fit_transform(X_train, y_train)
        X_test_filtered = selector.transform(X_test)
        return X_train_filtered, X_test_filtered


def create_new_features(data):
        """
        Applies some transformations or creates new features based on the given data.
        Returns modified_data.
        """
        # Example: Scaling numeric features using MinMaxScaler
        scaler = MinMaxScaler()
        numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_features] = scaler.fit_transform(data[numeric_features])

        # Example: Creating a new feature by combining existing features
        data['new_feature'] = data['feature1'] + data['feature2']

        return data


def fit_and_classify(X_train, y_train):
        """
        Fits a multilabel classification model on the training data and performs classification on the test data.
        Returns classification report.
        """
        model = OneVsRestClassifier(LogisticRegression())
        model.fit(X_train, y_train)
        return model


def plot_classification_report(report):
    """
    Plots a bar plot to visualize the classification report.
    """
    lines = report.split('\n')
    classes = []
    precision = []
    recall = []
    f1_score = []

    for line in lines[2:-3]:
        t = line.split()
        classes.append(t[0])
        precision.append(float(t[1]))
        recall.append(float(t[2]))
        f1_score.append(float(t[3]))

    x = range(len(classes))
    width = 0.2

    fig, ax = plt.subplots()
    ax.bar(x, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall', bottom=precision)
    ax.bar(x, f1_score, width, label='F1 Score', bottom=[i+j for i,j in zip(precision, recall)])

    ax.set_ylabel('Scores')
    ax.set_title('Classification Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()

    plt.show()


def main():
    # Step 1: Load the data
    file_path = "../data/test.feats.csv"
    data = load_data(file_path)

    numercial_data = filter_data(data)

    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=5)
    numercial_data = imputer.fit_transform(numercial_data)

    y_train = load_data("../data/train.labels.0.csv")

    # Step 5: Fit the classifier and perform classification
    model = fit_and_classify(numercial_data, y_train)
    y_predict = model.predict(numercial_data)
    r = classification_report(y_predict, y_train)
    print(r)

    # Step 6: Plot the classification report
    plot_classification_report(r)

if __name__ == "__main__":
    main()
