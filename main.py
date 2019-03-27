from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
import xlsxwriter
import InExcel

#
# #Load dataset
wine = datasets.load_wine()
# # Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)  # 70% training and 30% test

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

# Workbook() takes one, non-optional, argument
# which is the filename that we want to create.
workbook = xlsxwriter.Workbook('hello.xlsx')

# The workbook object is then used to add new
# worksheet via the add_worksheet() method.
worksheet = workbook.add_worksheet()
cell_format = workbook.add_format({'bold': True, 'font_color': 'red'})

# Start from the first cell.
# Rows and columns are zero indexed.
row = 0
column = 0

worksheet.write('A1', 'Test Data Values')
worksheet.write('B1', 'KNN')
worksheet.write('C1', 'Naive Bayes')
worksheet.write('D1', 'Decision tree')
worksheet.write('E1', 'PLA')


InExcel.writeArray(worksheet,y_test, knn_y_pred, 1)
InExcel.writeArray(worksheet,y_test,nb_y_pred,2)
# InExcel.writeArray(y_test,decision_tree_y_pred,3)
# InExcel.writeArray(y_test,ppn_y_pred,4)







#
# # iterating through content list
# row=1
# column=0
# for item in y_test:
#     worksheet.write(row, column, item)
#     row += 1
# column=1
# row = 1
# wrong_pred=0
# for item,i in zip(knn_y_pred,y_test):
#     if item!=i:
#         worksheet.write(row, column, item,cell_format)
#         row+=1
#         wrong_pred+=1
#     else:
#         worksheet.write(row, column, item)
#         row += 1
# worksheet.write(row, column, wrong_pred)
# column=2
# row = 1
# for item in nb_y_pred:
#     worksheet.write(row, column, item)
#     row += 1
# column=3
# row =1
# for item in decision_tree_y_pred:
#     worksheet.write(row, column, item)
#     row += 1
# column=4
# row = 1
# for item in ppn_y_pred:
#     worksheet.write(row, column, item)
#     row += 1
# row = 1
# workbook.close()