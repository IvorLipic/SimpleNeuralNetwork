# Simple Regression Neural Network with Genetic Algorithm

## CLI Arguments
```bash
--train path_to_train_set --test path_to_test_set --nn 5s --popsize 10 --elitism 1 --p 0.1 --K 0.1 --iter 10000
```


--nn -> defines hidden layers sizes, delimiter "s" (eg. 5s - 1 hidden layer with 5 neurons, 5s3s7 - 3 hidden layers with 5, 3 and 7 neurons respectively)   
--popsize -> population size (of NNs - hypothesis)  
--elitism -> number of best hypothesis from the last generation to be used in the new generation  
--p -> probability of mutation  
--K -> stdev of mutation noise  
--iter -> num of iterations  
