import numpy as np


class ExtraTreesClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            # Bagging: Randomly sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_subset, y_subset = X[indices], y[indices]

            # Build a decision tree
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_subset, y_subset)

            # Add the tree to the forest
            self.trees.append(tree)

    def predict(self, X):
        # Perform majority voting from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Return the most common class for each sample
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        unique_classes, counts = np.unique(y, return_counts=True)

        # Base case: If all samples have the same label or max depth is reached
        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return {'class': unique_classes[0]}

        # Base case: If the number of samples is less than the minimum required for a split
        if n_samples < self.min_samples_split:
            return {'class': unique_classes[np.argmax(counts)]}

        # Choose the best split based on Gini impurity
        split_index, split_value = self._find_best_split(X, y)

        # Base case: If no split improves Gini impurity
        if split_index is None:
            return {'class': unique_classes[np.argmax(counts)]}

        # Split the data and build the left and right subtrees
        left_mask = X[:, split_index] <= split_value
        right_mask = ~left_mask
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'index': split_index, 'value': split_value, 'left': left_tree, 'right': right_tree}

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None  # Cannot split

        # Calculate Gini impurity for the entire node
        current_counts = np.bincount(y)
        current_gini = 1.0 - sum((count / n_samples) ** 2 for count in current_counts)

        best_gini = float('inf')
        best_split_index = None
        best_split_value = None

        for feature_index in range(n_features):
            # Sort the feature values
            feature_values = np.unique(X[:, feature_index])
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2.0

            for threshold in thresholds:
                # Create binary masks for left and right splits
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                # Skip if one of the splits is empty
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate Gini impurity for the left and right nodes
                left_gini = self._calculate_gini_impurity(y[left_mask])
                right_gini = self._calculate_gini_impurity(y[right_mask])

                # Calculate the weighted sum of Gini impurities for the left and right nodes
                weighted_gini = (np.sum(left_mask) / n_samples) * left_gini + \
                                (np.sum(right_mask) / n_samples) * right_gini

                # Update the best split if the current split is better
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split_index = feature_index
                    best_split_value = threshold

        return best_split_index, best_split_value

    def _calculate_gini_impurity(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        gini_impurity = 1.0 - np.sum(probabilities ** 2)
        return gini_impurity

    def predict(self, X):
        if self.tree is None:
            raise ValueError("The tree has not been trained yet.")

        return np.apply_along_axis(self._predict_single, axis=1, arr=X)

    def _predict_single(self, sample):
        current_node = self.tree
        while 'class' not in current_node:
            if sample[current_node['index']] <= current_node['value']:
                current_node = current_node['left']
            else:
                current_node = current_node['right']
        return current_node['class']