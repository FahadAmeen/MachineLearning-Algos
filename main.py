from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import xlsxwriter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#
# #Load dataset
# wine = datasets.load_wine()

# print the dataset in the excel file
# wineDF = pd.DataFrame(data= np.c_[wine['data'], wine['target']],
#                      columns= wine['feature_names'] + ['target']).to_csv('Dataset.csv')

dataset = pd.read_csv("winequality_white.csv")
print (type(dataset))
X = dataset.iloc[:,:-1].values #Takes all rows of all columns except the last column
Y = dataset.iloc[:,-1].values # Takes all rows of the last column

# missing values
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
X = imp.fit_transform(X)
Y = Y.reshape(-1,1)
Y = imp.fit_transform(Y)
Y = Y.reshape(-1)

# labelencoder_X = LabelEncoder()
# X[:, -1] = labelencoder_X.fit_transform(X[:, -1])
# hotencoder0 = OneHotEncoder(ColumnTransformer = [0])
# hotencoder1 = OneHotEncoder(ColumnTransformer = [1])
# hotencoder2 = OneHotEncoder(ColumnTransformer = [2])
# hotencoder3 = OneHotEncoder(ColumnTransformer = [3])
# hotencoder4 = OneHotEncoder(ColumnTransformer = [4])
# hotencoder5 = OneHotEncoder(ColumnTransformer = [5])
# hotencoder6 = OneHotEncoder(ColumnTransformer = [6])
# hotencoder7 = OneHotEncoder(ColumnTransformer = [7])
# hotencoder8 = OneHotEncoder(ColumnTransformer = [8])
# hotencoder9 = OneHotEncoder(ColumnTransformer = [9])
# hotencoder10 = OneHotEncoder(ColumnTransformer = [10])
#
# X = hotencoder0.fit_transform(X).toarray()
# X = hotencoder1.fit_transform(X).toarray()
# X = hotencoder2.fit_transform(X).toarray()
# X = hotencoder3.fit_transform(X).toarray()
# X = hotencoder4.fit_transform(X).toarray()
# X = hotencoder5.fit_transform(X).toarray()
# X = hotencoder6.fit_transform(X).toarray()
# X = hotencoder7.fit_transform(X).toarray()
# X = hotencoder8.fit_transform(X).toarray()
# X = hotencoder9.fit_transform(X).toarray()
# X = hotencoder10.fit_transform(X).toarray()
#
# labelencoder_Y = LabelEncoder()
# Y = labelencoder_Y.fit_transform(Y)
#
# print (X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)  # 70% training and 30% test
# print (wine.target)


print ('Test results\n',y_test)


#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
knn_y_pred = knn.predict(X_test)
print('knn\n',knn_y_pred)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, knn_y_pred))

gnb = GaussianNB()

nb_y_pred = gnb.fit(X_train, y_train).predict(X_test)
print('naive bayes\n',nb_y_pred)
print("Accuracy:", metrics.accuracy_score(y_test, nb_y_pred))

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
decision_tree_y_pred = clf.predict(X_test)
print('Decision tree\n',decision_tree_y_pred)
print("Accuracy:", metrics.accuracy_score(y_test, decision_tree_y_pred))


ppn = Perceptron(max_iter=40, tol=0.1, random_state=0)
ppn=ppn.fit(X_train, y_train)
ppn_y_pred = ppn.predict(X_test)
print('Perceptron\n',ppn_y_pred)
print("Accuracy:", metrics.accuracy_score(y_test, ppn_y_pred))


# ############################################
# Writing in xlsx file named Classifier.xlsx #
# ############################################

workbook = xlsxwriter.Workbook('Classifiers.xlsx')
worksheet = workbook.add_worksheet('classifiers')
cell_format = workbook.add_format({'bold': True, 'font_color': 'red'})
cell_format_sum = workbook.add_format({'bold': True, 'font_color': 'blue'})
cell_format_accuracy = workbook.add_format({'bold': True})

# Start from the first cell.
# Rows and columns are zero indexed.
row = 0
column = 0

worksheet.write('A1', 'Test Data Values',cell_format_accuracy)
worksheet.write('B1', 'KNN',cell_format_accuracy)
worksheet.write('C1', 'NB',cell_format_accuracy)
worksheet.write('D1', 'DT',cell_format_accuracy)
worksheet.write('E1', 'PLA',cell_format_accuracy)

#
# iterating through content list
row=1
column=0
for item in y_test:
    worksheet.write(row, column, item,cell_format_accuracy)
    row += 1
column=1
row = 1
wrong_pred=0
for item,i in zip(knn_y_pred,y_test):
    if item!=i:
        worksheet.write(row, column, item,cell_format)
        row+=1
        wrong_pred+=1
    else:
        worksheet.write(row, column, item)
        row += 1
worksheet.write(row, column, wrong_pred,cell_format_sum)


column=2
row = 1
wrong_pred=0
for item,i in zip(nb_y_pred,y_test):
    if item!=i:
        worksheet.write(row, column, item,cell_format)
        row+=1
        wrong_pred+=1
    else:
        worksheet.write(row, column, item)
        row += 1
worksheet.write(row, column, wrong_pred,cell_format_sum)



column=3
row =1
wrong_pred=0
for item,i in zip(decision_tree_y_pred,y_test):
    if item!=i:
        worksheet.write(row, column, item,cell_format)
        row+=1
        wrong_pred+=1
    else:
        worksheet.write(row, column, item)
        row += 1
worksheet.write(row, column, wrong_pred,cell_format_sum)


column=4
row = 1
wrong_pred=0
for item,i in zip(ppn_y_pred,y_test):
    if item!=i:
        worksheet.write(row, column, item,cell_format)
        row+=1
        wrong_pred+=1
    else:
        worksheet.write(row, column, item)
        row += 1
worksheet.write(row, column, wrong_pred,cell_format_sum)

# Wrong predictions tag
row +=1
column =0
worksheet.write(row, column, 'Wrong predictions', cell_format_accuracy)

# Accuracy in xls
row +=1
column =0
worksheet.write(row, column, 'Accuracy', cell_format_accuracy)

column =1
worksheet.write(row, column, metrics.accuracy_score(y_test, knn_y_pred))

column =2
worksheet.write(row, column, metrics.accuracy_score(y_test, nb_y_pred))

column =3
worksheet.write(row, column,metrics.accuracy_score(y_test, decision_tree_y_pred) )

column =4
worksheet.write(row, column,metrics.accuracy_score(y_test, ppn_y_pred) )


workbook.close()


############################
# ploting confusion matrix #
############################

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, knn_y_pred, classes=wine.target_names,
#                       title='Confusion matrix KNN')
# plot_confusion_matrix(y_test, nb_y_pred, classes=wine.target_names,
#                       title='Confusion matrix NB')
# plot_confusion_matrix(y_test, decision_tree_y_pred, classes=wine.target_names,
#                       title='Confusion matrix DT')
# plot_confusion_matrix(y_test, ppn_y_pred, classes=wine.target_names,
#                       title='Confusion matrix PLA')

knn_fig='knn.png'
nb_fig='nb.png'
dt_fig='dt.png'
ppn_fig='ppn.png'

plt.savefig(knn_fig)
plt.savefig(nb_fig)
plt.savefig(dt_fig)
plt.savefig(ppn_fig)
plt.show()

