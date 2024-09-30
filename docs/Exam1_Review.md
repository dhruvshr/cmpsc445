# CMPSC445 Applied ML Exam 1 Review

1. ML Concepts and Algorithms:
    - Main characteristics of ML and relation to data science.

        - Data science is about extracting insights from messy (structured and unstructured) data.
        - The data science methodology:
            - Understanding a real-world problem
            - Analysis of data requirements
            - Collection of data
            - Processing of Data
            - Use of ML
            - Evaluation of ML models
            - Constant Improvement

        - ML is a tool for exploring, representing, and learning from the data to create a model that can either classify or predict outcomes based on the input features of the dataset.
    
    - ML Categories: 
        - Supervised Learning vs. Unsupervised Learning
            - Supervised Learning:
                - Given a set of data points associated to a set of outcomes, we want to build a classifier that learns how to predict the targets from the features.
                - Regression -> Continous (e.x. Linear Regression)
                - Classification -> Discrete (Class) (e.x. - Logistic Regression, SVM, NB)

                - Discriminative Models
                    - Directly estimate P(y|x) <- Decision Boundary (e.g. Regressions, SVMs)
                - Generative Models
                    - Estimate P(x|y) to then deduce P(y|x) <- Probability Distributions of Data (e.g. GDA, NB)

                - Applications:
                    - Classification: categories
                    - Regression: predictions
                    - Predicting Decisions: decision trees 

            - Unsupervised Learning:
                - Give a set of unlabeled data, we want to find the hidden patterns without prior knowledge of the outcomes.
                - E.x. K-Means Clustering, PCA

                - Harder to evaluate as there are no predefined labels; often relies on metrics like silhouette coefficient. 
                
                - Applications:
                    - Clustering: grouping similar data points together
                    - Anomaly Detection: identifying unusual data points.
                    - Dimensionality Reduction: Simplifying data while retaining its structure.
                        - Principal Component Analysis
                            - Eigenvalue, Eigenvectors

    - Data feature (attribute), dimension, label 
        - Data features
            - An individual measurable property or characteristic of a data point.
            - Feautres are the inputs that the model uses to learn patterns.
        - Dimensions
            - Refers to the number of features in a dataset.
            - Each additional feature adds a dimension, resulting in a multi-dimensional space where data points exist.
        - Labels
            - An output or target variable in supervised learning.
            - Model uses labels to predict results based on the input features.

    - Reasons for separating training data and test data; Reasons for performing cross-validation in training;
        - Data separation into training and testing data:
            - Overfitting: seperating data helps ensure that the model generalizes its understanding to unseen data rather than just memorizing the data.
            - Performance evaluation: The test data (ideally) provides an unbiased assessment of the model's performance after training, indicating how well it performs in real-world or unfamiliar situations.
            - Model Tuning: It helps to fine-tune the hyperparameters based on the training data without influencing the evaluaution process.

        - Performing Cross Validation:
            - Dynamic Training of data: Cross-Validation uses multiple subsets of the data for training and test, maximizing the amount of data used for training while still evaluating performance.
            - Reduced Variablitity: It provides a more reliable estimate of the model by averaging results over different folds, reducing the influence a single split.
            - Model robustness: it helps identify how well the model generalizes to different data distribtuions, leading to better model selection and hyperparameter tuning.

    - Supervised Learning Algorithms
        - Linear Regression: `

                
