# Data Cleaning

## How to prepare the data

> Garbage in, Garbage out!

1. Cleaning
    - `pd.read_csv()`
    - dealing with missing value
        - replace with the mean price
    - remove them
2. Transformation
    - `dataFrame.drop()`
    - normalization `sklearn.preprocessing.StandardScalar`
3. Reduction
    - PCA: 
        1. Normalize
        2. Compute Covariance Matrix
        3. Eigen Decomposition
        4. `sklearn.decomposition.PCA`
    - T-SNE
        - Best for visualization
        - [Tutorial](https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm)
            - Reading N.A.
        - [Tutorial 2](https://www.quora.com/What-advantages-the-t-sne-algorithm-has-over-pca)
            - Reading N.A.
    - LDA:  (Linear Discriminant Analysis) -- For the supervised 
        - calculate the mean vector for each class
    
> Architecture Engineering IS the new feature Engineering

## Resource

- [Udacity Pandas Course](Intro to Data Analysis course)
- [Udacity Data Cleaning Course](Data Wrangling course)
- [Udacity Intro to ML Course](https://www.udacity.com/course/intro-to-machine-learning--ud120)