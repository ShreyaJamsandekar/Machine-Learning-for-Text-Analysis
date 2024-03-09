import argparse
#import nltk
#nltk.download('averaged_perceptron_tagger')

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

#import tensorflow as tf

class EOSClassifier:
    def train(self, trainX, trainY):

        self.abbrevs = load_wordlist('classes/abbrevs')
        self.sentence_internal = load_wordlist("classes/sentence_internal")
        self.timeterms = load_wordlist("classes/timeterms")
        self.titles = load_wordlist("classes/titles")
        self.unlikely_proper_nouns = load_wordlist("classes/unlikely_proper_nouns")

        self.stop_words = load_wordlist("classes/stop_words")


        # In this part of the code, we're loading a Scikit-Learn model.
        # We're using a DecisionTreeClassifier... it's simple and lets you
        # focus on building good features.
        self.clf = DecisionTreeClassifier()  # TODO: experiment with different models
        #self.clf = SVC(kernel='sigmoid')
        #self.clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        self.clf = RandomForestClassifier(n_estimators=100)


        X = [self.extract_features(x) for x in trainX]
        self.clf.fit(X, trainY)

    def extract_features(self, array):
        id, word_m3, word_m2, word_m1, period, word_p1, word_p2, word_3, left_reliable, right_reliable, num_spaces = array

        # The "features" array holds a list of
        # values that should act as predictors.
        # We want to take some component(s) above and "translate" them to a numerical value.
        # For example, our 4th feature has a value of 1 if word_m1 is an abbreviation,
        # and 0 if not.

        features = [  # TODO: add features here
            left_reliable,
            right_reliable,
            num_spaces,
            len(word_m1),
            len(word_m2),
            len(word_m3),
            len(word_p1),
            len(word_p2),
            len(word_3),
            1 if word_m1 in self.abbrevs else 0,
            1 if word_m1 in self.titles else 0,    
            1 if word_p1.isupper() else 0,
            1 if word_m2 in self.sentence_internal else 0, 
            1 if word_p2 in self.sentence_internal else 0,
            1 if word_m3 in self.timeterms else 0,
            1 if word_p1 in self.unlikely_proper_nouns else 0,
            1 if word_m3 in self.sentence_internal else 0,
            #1 if word_p1 == '.' else 0,  # Is the next word a period?
            1 if word_p1 == ',' else 0,  # Is the next word a comma?
            1 if word_m1.endswith('.') else 0,  # Does the previous word end with a period?

            1 if word_m1.endswith(('.', ',', '!', '?')) else 0,  # Is the preceding word ending with punctuation?
            1 if word_p1[0].isupper() else 0,  # Is the following word capitalized?
            1 if word_m1.isdigit() else 0,  # Is the preceding word a number?
            1 if word_p1.isdigit() else 0,  # Is the following word a number?
         
            1 if word_p1 in self.abbrevs else 0,  # Is the following word an abbreviation?
            1 if word_m1.lower() in self.stop_words else 0,  # Is the preceding word a stop word?
            1 if word_p1.lower() in self.stop_words else 0,  # Is the following word a stop word?
            1 if word_m1.istitle() else 0,  # Is the preceding word a title?
            1 if word_p1.istitle() else 0,  # Is the following word a title?
            1 if word_m1 in self.timeterms else 0,  # Is the preceding word a known time-related term?
            1 if word_p1 in self.timeterms else 0,  # Is the following word a known time-related term?
            1 if word_m1.endswith(('.', ',', '!', '?')) else 0,  # Is the preceding word ending with punctuation?

            1 if word_m1.isdigit() else 0,  # Is the preceding word a number?
            1 if word_p1.isdigit() else 0,  # Is the following word a number?

            1 if word_m1.lower() in self.stop_words else 0,  # Is the preceding word a stop word?
            1 if word_p1.lower() in self.stop_words else 0,  # Is the following word a stop word?

            1 if word_m1.istitle() else 0,  # Is the preceding word a title?
            1 if word_p1.istitle() else 0,  # Is the following word a title?

                
        ]
   

        
        return features

    def classify(self, testX):
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)


def load_wordlist(file):
    with open(file) as fin:
        return set([x.strip() for x in fin.readlines()])


def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split()
            X.append(arr[1:])
            y.append(arr[0])
        return X, y


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
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()
    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)

    classifier = EOSClassifier()
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