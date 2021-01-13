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

    # YOUR CODE HERE
    if len(y) == 0:
        return 0.
    
    pk = np.mean(y, axis=0)
    
    return - np.sum(pk * np.log(pk + EPS))
    
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
    # YOUR CODE HERE
    if len(y) == 0:
        return 0.
    
    pk = np.mean(y, axis=0)
    
    return 1 - np.sum(pk * pk)
    
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
    
    # YOUR CODE HERE
    if len(y) == 0:
        return 0.
    
    return np.var(y)

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

    # YOUR CODE HERE
    if len(y) == 0:
        return 0.
    
    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]

# new function to predict value for classification
def predicted_class(y):
    return np.mean(y, axis=0).argmax()

class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0.0):
        self.feature_index = feature_index
        self.threshold = threshold
   
        self.left_child = None
        self.right_child = None
        
        # For when the node is a leaf
        self.value = None
        self.proba = proba
        
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
        
        # Let define the criterion function from the get go
        self.H = self.all_criterions[criterion_name][0]
        
        # Define the prediction method
        self.predicted_values = None
        if self.criterion_name == 'variance':
            self.predicted_values = np.mean
        elif self.criterion_name in 'mad_median':
            self.predicted_values = np.median
        else:
            self.predicted_values = predicted_class
            
        
        
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

        # YOUR CODE HERE
        ind = (X_subset[:, feature_index] < threshold)
        
        X_left, y_left = X_subset[ind], y_subset[ind]
        X_right, y_right = X_subset[~ind], y_subset[~ind]
        
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

        # YOUR CODE HERE
        ind = (X_subset[:, feature_index] < threshold)

        y_left = y_subset[ind]
        y_right = y_subset[~ind]
        
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
        # YOUR CODE HERE
        feature_index = None
        threshold = None
        best_G = np.inf
        N = len(X_subset)
        
        for current_feature in range(X_subset.shape[1]):
            thresholds = np.unique(X_subset[:, current_feature])
            
            for t in thresholds:
                y_left, y_right = self.make_split_only_y(current_feature, t, X_subset, y_subset)
                H_L = self.H(y_left)
                H_R = self.H(y_right)
                
                G = (len(y_left) / N) * H_L + (len(y_right) / N) * H_R
                
                if G < best_G:
                    best_G = G
                    feature_index = current_feature
                    threshold = t
                    
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
        
        # YOUR CODE HERE
        self.depth += 1
        if self.depth < self.max_depth and X_subset.shape[0] >= self.min_samples_split:
            
            best_feature, best_threshold = self.choose_best_split(X_subset, y_subset)
            new_node = Node(best_feature, best_threshold)
            
            left_child, right_child = self.make_split(best_feature, best_threshold, X_subset, y_subset)
            new_node.left_child = self.make_tree(*left_child)
            new_node.right_child = self.make_tree(*right_child)
            
        else: # we have a leaf
            new_node = Node(-1, -1) # We flag leaf nodes by setting feature_index and threshold to -1
            new_node.value = self.predicted_values(y_subset)
            
            if self.classification:
                new_node.proba = np.mean(y_subset, axis=0)
        
        # We reduce the depth to compensate for the two calls to self.depth += 1 we make on
        # the same level for left_child and right_child.
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
    
    
    
    def pred_traversal(self, node, X, ind, predictions, return_probas):
        
        # We use the flag to check whether we are in a leaf node
        if node.feature_index != -1:
            left, right = self.make_split(node.feature_index, node.threshold, X, ind)
            self.pred_traversal(node.left_child, *left, predictions, return_probas)
            self.pred_traversal(node.right_child, *right, predictions, return_probas)
        elif return_probas:
            predictions[ind] = node.proba
        else:
            predictions[ind] = node.value
    
    
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

        # YOUR CODE HERE
        y_predicted = np.zeros((X.shape[0], 1))
        ind = np.arange(X.shape[0])
        self.pred_traversal(self.root, X, ind, y_predicted, return_probas=False)
        
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

        # YOUR CODE HERE
        y_predicted_probs = np.zeros((X.shape[0], self.n_classes))
        ind = np.arange(X.shape[0])
        self.pred_traversal(self.root, X, ind, y_predicted_probs, return_probas=True)
        
        return y_predicted_probs
