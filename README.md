# Entity Resolution
Matching algorithm for movies in Amazon and Rotten Tomatoes datasets

**Task**

Given two datasets (one from Amazon and one from Rotten Tomatoes) that describe the same movie entities, identify in the test set whether each pair of two IDs (one ID each from the two datasets) refer to the same movie. Write a script that loads these datasets, runs a binary classificiation matching algorithm to determine if two paired movies are the same, and then measure the precision, recall, and F1-score of the algorithm on the training set. Final predictions are submitted to a leaderboard page on Instabase for competition among students in groups of 1-3.

This assignment was done for the [Spring 2017 course in Computing Systems for Data Science (COMS W4121)](https://w4121.github.io/).

**Results**

The final matching algorithm involved standardizing features involving movie time length; cleaning the time and movie star variables;  engineering new features that computed string similarity scores for corresponding pairs of the movie star and movie director variables, as well as calculated the film length difference of each movie pair; and running a random forest classifier with optimized parameters based on these new features. On the Instabase leaderboard, this model resulted in a precision, recall, and F-measure of 91.25%.
