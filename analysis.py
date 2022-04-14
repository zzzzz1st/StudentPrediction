import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
# ADA BOOST AND PERCEPTRON ANALYSIS ON 4 BINARY TYPES
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def analysis(df):
    y = df["final_result"]
    x = df.drop(columns="final_result")
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    weight = x_train["weight"]
    x_train = x_train.drop(columns="weight")
    x_test = x_test.drop(columns="weight")
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(x_train, y_train, sample_weight=weight)
    y_predicted = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    loss = 1 - accuracy
    print("Ada boost Accuracy : " + str(accuracy))
    print("Ada boost Loss : " + str(loss))
    print("Ada boost Precision : " + str(precision_score(y_test, y_predicted)))
    print("Ada boost Recall : " + str(recall_score(y_test, y_predicted)))
    clf = MLPClassifier()
    clf.fit(x_train, y_train)
    y_predicted = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    loss = 1 - accuracy
    print("MLP Accuracy : " + str(accuracy))
    print("MLP Loss : " + str(loss))
    print("MLP Precision : " + str(precision_score(y_test, y_predicted)))
    print("MLP Recall : " + str(recall_score(y_test, y_predicted)))


studentpf = pd.read_pickle("studentpf.pkl")
studentdf = pd.read_pickle("studentdf.pkl")
studentdp = pd.read_pickle("studentdp.pkl")
studentwp = pd.read_pickle("studentwp.pkl")

print(studentpf["final_result"].value_counts())
print(studentdf["final_result"].value_counts())
print(studentdp["final_result"].value_counts())
print(studentwp["final_result"].value_counts())

analysis(studentpf)
analysis(studentdf)
analysis(studentdp)
analysis(studentwp)
