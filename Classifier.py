#import our datasest from sklearn
from sklearn import datasets,metrics

# Load in the `digits` data
digits = datasets.load_digits()

# Print the `digits` data
print(digits.data)

#import our Classifier avalaible
from sklearn.svm import SVC
# Create a classifier: a support vector classifier
classifier = SVC(gamma=0.001)



# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))





# We train our model with the first half of data
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

#Now we test our model by predicting the second half of our data
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))



