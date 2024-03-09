import argparse

from itertools import groupby
from sklearn import linear_model, naive_bayes, tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer
from sklearn import naive_bayes, linear_model, tree, svm 


class SegmentClassifier:
    def train(self, trainX, trainY):
        # self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Use TF-IDF vectorization
        # X_train_tfidf = self.tfidf_vectorizer.fit_transform(trainX)
        # self.clf = SVC(kernel="linear")
        # self.clf = linear_model.LogisticRegression()
        self.clf = RandomForestClassifier(n_estimators=1000)
        # self.clf = naive_bayes.MultinomialNB() # TODO: experiment with different models
        X = [self.extract_features(x) for x in trainX]
        self.clf.fit(X, trainY)
        #self.clf.fit(X_train_tfidf, trainY)

    def extract_features(self, text):
        words = text.split()
        # print(words)
        features = [  # TODO: add features here
            len(text),
            len(text.strip()),
            len(words),           
            text.count(' '),
            sum(1 if '*' in word else 0 for word in words),
            sum(1 if 'o' in word else 0 for word in words),
            sum(1 if '0' in word else 0 for word in words),
            sum(1 if '+' in word else 0 for word in words),
            sum(1 if '-' in word else 0 for word in words),
            sum(1 if '/' in word else 0 for word in words),
            sum(1 if '>' in word else 0 for word in words),
            sum(1 if ':' in word else 0 for word in words),
            sum(1 if '"' in word else 0 for word in words),
            sum(1 if '	' in word else 0 for word in words),
            sum(1 if '|' in word else 0 for word in words),
            sum(1 if '-' in word else 0 for word in words),
            sum(1 if '(' in word else 0 for word in words),
            sum(1 if '\\' in word else 0 for word in words),
            sum(1 if '_' in word else 0 for word in words),
            sum(1 if '[' in word else 0 for word in words),
            sum(1 if '=' in word else 0 for word in words),
            sum(1 if '@' in word else 0 for word in words),
            sum(1 if '~' in word else 0 for word in words),
            sum(1 if '!' in word else 0 for word in words),
            sum(1 if '%' in word else 0 for word in words),
            sum(1 if '^' in word else 0 for word in words),
            sum(1 if ')' in word else 0 for word in words),
            sum(1 if '&' in word else 0 for word in words),
            sum(1 if '#' in word else 0 for word in words),
            sum(1 if '{' in word else 0 for word in words),
            sum(1 if '?' in word else 0 for word in words),
            sum(1 if '$' in word else 0 for word in words),
            sum(1 if w.isupper() else 0 for w in words),
            sum(1 if text.islower() else 0 for text in words),
            sum(1 if text.isdigit() else 0 for text in words),
            sum(1 if text.isspace() else 0 for text in words)
        ]
        return features

    def classify(self, testX):
        # X_test_tfidf = self.tfidf_vectorizer.transform(testX) # TF_IDF for test data
        # return self.clf.predict(X_test_tfidf)
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)


def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split('\t', 1)
            if arr[0] == '#BLANK#':
                continue
            X.append(arr[1])
            y.append(arr[0])
        return X, y


def lines2segments(trainX, trainY):
    segX = []
    segY = []
    for y, group in groupby(zip(trainX, trainY), key=lambda x: x[1]):
        if y == '#BLANK#':
            continue
        x = '\n'.join(line[0].rstrip('\n') for line in group)
        segX.append(x)
        segY.append(y)
    return segX, segY


def evaluate(outputs, golds):
    correct = 0
    for h, y in zip(outputs, golds):
        if h == y:
            correct += 1
    print(f'{correct} / {len(golds)}  {correct / len(golds)}')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--format', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()

    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)

    if args.format == 'segment':
        trainX, trainY = lines2segments(trainX, trainY)
        testX, testY = lines2segments(testX, testY)

    classifier = SegmentClassifier()
    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX)

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    if args.errors is not None:
        with open(args.errors, 'w') as fout:
            for y, h, x in zip(testY, outputs, testX):
                if y != h:
                    print(y, h, x, sep='\t', file=fout)

    if args.report:
        print(classification_report(testY, outputs))
    else:
        evaluate(outputs, testY)


if __name__ == '__main__':
    main()