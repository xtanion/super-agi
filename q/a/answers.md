3. You have `m` training examples and `n` features. Your feature vectors are however sparse and average number of non-zero entries in each train example is `k` and `k` << `n`. What is the approximate computational cost of each gradient descent iteration of logistic regression in modern well written packages?

Ans: To compute the dot product of the feature vector with the parameter vector. Since the average number of non-zero entries in each training example is `k`, this operation has a complexity of `
O(k)`

