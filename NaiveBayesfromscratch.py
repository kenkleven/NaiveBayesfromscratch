from collections import defaultdict, Counter
import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}
        self.classes = set()

    def fit(self, X, y):
        self.classes = set(y)
        total_samples = len(y)
        label_count = Counter(y)

        self.class_probs = {cls: count / total_samples for cls, count in label_count.items()}

        self.feature_probs = {cls: defaultdict(lambda: defaultdict(int)) for cls in self.classes}

        for x_vec, label in zip(X, y):
            for idx, val in enumerate(x_vec):
                self.feature_probs[label][idx][val] += 1

        for cls in self.classes:
            for idx in self.feature_probs[cls]:
                total = sum(self.feature_probs[cls][idx].values())
                for val in self.feature_probs[cls][idx]:
                    self.feature_probs[cls][idx][val] /= total

    def predict(self, X):
        preds = []
        for x_vec in X:
            class_scores = {}
            for cls in self.classes:
                prob = np.log(self.class_probs[cls])
                for idx, val in enumerate(x_vec):
                    prob += np.log(self.feature_probs[cls][idx].get(val, 1e-6))  
                class_scores[cls] = prob
            preds.append(max(class_scores, key=class_scores.get))
        return preds
    
    def accuracy(self, y_true, y_pred):
        correct = np.sum(np.array(y_true) == np.array(y_pred))
        return correct / len(y_true)
    
    def precision(self, y_true, y_pred, positive_class=1):
        tp = 0
        fp = 0 
        
        for yt, yp in zip(y_true, y_pred):
            if yp == positive_class:
                if yt == positive_class:
                    tp += 1
                else:
                    fp += 1
        
        if tp + fp == 0:
            return 0.0 
        return tp / (tp + fp)
    
    def recall(self, y_true, y_pred, positive_class=1):
        tp = 0
        fn = 0 
        
        for yt, yp in zip(y_true, y_pred):
            if yt == positive_class:
                if yp == positive_class:
                    tp += 1
                else:
                    fn += 1
        
        if tp + fn == 0:
            return 0.0 
        return tp / (tp + fn)
    
    def f1_score(self, y_true, y_pred, positive_class=1):
        prec = self.precision(y_true, y_pred, positive_class)
        rec = self.recall(y_true, y_pred, positive_class)
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)
    
    
    

class NaiveBayesRegressor:
    def __init__(self):
        self.means = None
        self.vars = None
        self.y = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X = np.array(X)
        self.y = np.array(y)
        self.means = self.X.mean(axis=0)
        self.vars = self.X.var(axis=0) + 1e-6
        
    def gaussian_prob(self, x):
        exponent = -((x - self.means) ** 2) / (2 * self.vars)
        return np.exp(exponent) / np.sqrt(2 * np.pi * self.vars)

    def predict(self, X):
        X = np.array(X)
        preds = []
        for x in X:
            probs = []
            for x_train in self.X:
                exponent = -((x - x_train) ** 2) / (2 * self.vars)
                p = np.exp(exponent) / np.sqrt(2 * np.pi * self.vars)
                probs.append(np.prod(p))  
            probs = np.array(probs)
            probs += 1e-6 
            weights = probs / np.sum(probs)
            preds.append(np.sum(self.y * weights))
        return np.array(preds)


    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)