import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    probas = np.mean(y, axis=0)
    return - np.sum(probas * np.log(probas + EPS))
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    probas = np.mean(y, axis=0)
    return 1 - np.sum(probas ** 2)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    return np.mean((y - np.mean(y)) ** 2)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]



class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """
        right_ind = []
        left_ind = []
        for i in range(X_subset.shape[0]):
            if X_subset[i][feature_index] < threshold:
                left_ind.append(i)
            else:
                right_ind.append(i)
        X_left = X_subset[left_ind]
        X_right = X_subset[right_ind]
        y_left = y_subset[left_ind]
        y_right = y_subset[right_ind]
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """
        right_ind = []
        left_ind = []
        for i in range(X_subset.shape[0]):
            if X_subset[i][feature_index] < threshold:
                left_ind.append(i)
            else:
                right_ind.append(i)
        y_left = y_subset[left_ind]
        y_right = y_subset[right_ind]
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        feature_index = 0
        threshold = 0
        H = self.criterion
        size = X_subset.shape[0]
        N = size
        G = np.inf
        for i in range(X_subset.shape[1]):
            threshold_s = np.sort(np.unique(X_subset[:, i]))
            for t in threshold_s[1:-1]:
                left, right = self.make_split_only_y(i, t, X_subset, y_subset)
                G_new = left.shape[0] / size * H(left) + right.shape[0] / size * H(right)
                if G_new < G:
                    feature_index = i
                    threshold = t
                    G = G_new
        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        print(X_subset.shape)
        if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
            f = lambda y : np.argmax(np.sum(y, axis=0))
            p = lambda y : np.mean(y, axis=0)
        elif self.criterion_name == 'mad_median':
            f = lambda y: np.median(y)
            p = lambda y: np.mean(y)
        elif self.criterion_name == 'variance':
            f = lambda y: np.mean(y)
            p = lambda y: np.mean(y)
        min_leaves = X_subset.shape[0]
        self.depth += 1
        if self.depth < self.max_depth and min_leaves >= self.min_samples_split:
            feature_ind, threshold = self.choose_best_split(X_subset, y_subset)
            new_node = Node(feature_ind, threshold)
            left, right = self.make_split(feature_ind, threshold, X_subset, y_subset)
            new_node.left_child = self.make_tree(left[0], left[1])
            new_node.right_child = self.make_tree(right[0], right[1])
        else:
            new_node = Node(0, 0)
            new_node.value = f(y_subset)
            new_node.proba = p(y_subset)
            print(new_node.proba)
        self.depth -= 1
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def rec_pred(self, X, indices, node, pred, proba=False):
        if node.left_child is None:
            for i in indices:
                if not proba:
                    pred[i] = node.value
                else:
                    pred[i] = node.proba
        else:
            left, right = self.make_split(node.feature_index, node.value, X, indices)
            self.rec_pred(left[0], left[1], node.left_child, pred, proba)
            self.rec_pred(right[0], right[1], node.right_child, pred, proba)

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        y_predicted = np.zeros(X.shape[0])
        prediction = {}
        indices = np.arange(X.shape[0])
        tree = self.root
        self.rec_pred(X, indices, tree, prediction)
        for ind, value in prediction.items():
            y_predicted[ind] = value
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'
        y_predicted_probs = np.zeros((X.shape[0], self.n_classes))
        prediction = {}
        indices = np.arange(X.shape[0])
        tree = self.root
        self.rec_pred(X, indices, tree, prediction, True)
        for ind, proba in prediction.items():
            y_predicted_probs[ind] = proba
        return y_predicted_probs