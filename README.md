# Make a ExtraTreesClassifier from scratch 🧩

This project provides an implementation of a Extra Trees Classifier algorithm from scratch, without libraries scikit-learn. <br>
i'll embark on a journey to create an Extra Trees Classifier from scratch, immersing ourselves in the intricate details of its inner workings.

By constructing each component step-by-step, we'll uncover the elegance and effectiveness of this ensemble learning technique. Our journey begins with crafting a clear pseudocode, ensuring a well-structured and efficient implementation.

Let's dive into the core algorithm 🚀

## Pseudocode for fitting model <br>
Input: n_estimators, max_depth, min_samples_split <br>
Output: Initialized ETC model <br>
```
1. Set n_estimators, max_depth, dan min_samples_split sesuai parameter input.
2. Initialize the array to store the decision trees.
3. Return the initialized Extra Trees model.
```

## Pseudocode for training model <br>
Input: Training data (X_train, y_train) <br>
Output: Ensemble of decision trees <br>
```
1. Loop as many times as n_estimators:
    a. Take a random sample with replacement from the training data.
    b. Build a decision tree using the subset of data taken.
    c. Add the decision tree to the ensemble.
2. Return the ensemble of decision trees.
```

## Pseudocode for predict data <br>
Input: Test data (X_test) <br>
Output: Class prediction for each sample in X_test <br>
```
1. Loop for each sample in X_test:
    a. Perform prediction using each tree in the ensemble.
    b. Collect the prediction results from all trees.
2. Perform majority voting for each sample:
    a. Calculate the frequency of each class based on the prediction results.
    b. Select the class with the highest frequency as the final prediction.
3. Return the class prediction for each sample in X_test.
```

## How to run?
- clone the repo
- take the dummy dataset or the dataset you want to train on.
- import package with command -> `from ExtraTree import ExtraTreesClassifier`
- Example usage :
    ```
    # Training
    etc = ExtraTreesClassifier( n_estimators=1, max_depth=2, min_samples_split=5)
    etc.fit(X_train, y_train)
    # predict
    predictions = etc.predict(X_test)
    ```

#### Acknowledgements
- https://julienbeaulieu.gitbook.io/wiki/sciences/machine-learning/decision-trees
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#:~:text=An%20extra%2Dtrees%20classifier.,accuracy%20and%20control%20over%2Dfitting.
