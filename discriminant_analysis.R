Comp_priors <- function(train_labels) {
  #' Compute the priors of each class label 
  #' 
  #' @param train_labels a vector of labels with length equal to n
  #' @return a probability vector of length K = 10
  
  
  K <- 10
  pi_vec <- rep(0, K)
  
  sum_table = table(train_labels)
  
  # Get the sum of the number of each class
  sum_values = as.numeric(sum_table)
  
  n = length(train_labels)
  for (i in 0:9){
    pi_vec[i+1] = sum_values[i+1]/n
  }
  
  return(pi_vec)
}



Comp_cond_means <- function(train_data, train_labels) {
  #' Compute the conditional means of each class 
  #' 
  #' @param train_data a n by p matrix containing p features of n training points
  #' @param train_labels a vector of labels with length equal to n
  #' 
  #' @return a p by 10 matrix, each column represents the conditional mean given
  #'   each class.
  
  K <- 10
  p <- ncol(train_data)
  mean_mat <- matrix(0, p, K)
  
  for (i in 1:length(train_labels)){
    mean_mat[,train_labels[i]+1] = mean_mat[,train_labels[i]+1] + train_data[i,]
  }
  
  # Divide by number of each category (700 total for each k)
  mean_mat = mean_mat/(length(train_labels)/K)
  
  return(mean_mat)
}



Comp_cond_covs <- function(train_data, train_labels, cov_equal = FALSE) {
  #' Compute the conditional covariance matrix of each class
  #' 
  #' @param train_data a n by p matrix containing p features of n training points
  #' @param train_labels a vector of labels with length equal to n
  #' @param cov_equal TRUE if all conditional covariance matrices are equal, 
  #'   otherwise, FALSE 
  #' 
  #' @return 
  #'  if \code{cov_equal} is FALSE, return an array with dimension (p, p, K),
  #'    containing p by p covariance matrices of each class;
  #'  else, return a p by p covariance matrix. 
  
  K <- 10
  p <- ncol(train_data)
  
  mean_matrix = Comp_cond_means(train_data, train_labels)
  n = length(train_labels)
  
  if (cov_equal == FALSE){
    # return an array with dimension (p, p, K), containing p by p covariance matrices of each class
    # for QDA
    
    result = array(0, dim = c(p, p, K))
    
    for (k in 0:9){
      for (i in 1:n){
        if (train_labels[i] == k){
          diff_mean = train_data[i,] - mean_matrix[,k+1]
          result[,,k+1] = result[,,k+1] + matrix(diff_mean) %*% t(matrix(diff_mean))
        }
      }
    }
    
    cov_arr <- result/(n/K - 1)
    
  } else {
    # return a p by p covariance matrix
    # for LDA
    
    result = matrix(0, p, p)
    
    for (k in 0:9){
      for (i in 1:n){
        if (train_labels[i] == k){
          diff_mean = train_data[i,] - mean_matrix[,k+1]
          result = result + matrix(diff_mean) %*% t(matrix(diff_mean))
        }
      }
    }
    
    cov_arr <- result/(n-K)
  }
  return(cov_arr)
}



Predict_posterior <- function(test_data, priors, means, covs, cov_equal) {
  
  #' Predict the posterior probabilities of each class 
  #'
  #' @param test_data a n_test by p feature matrix 
  #' @param priors a vector of prior probabilities with length equal to K
  #' @param means a p by K matrix containing conditional means given each class
  #' @param covs covariance matrices of each class, depending on \code{cov_equal}
  #' @param cov_equal TRUE if all conditional covariance matrices are equal; 
  #'   otherwise FALSE.
  #'   
  #' @return a n_test by K matrix: each row contains the posterior probabilities 
  #'   of each class.
  
  n_test <- nrow(test_data)
  K <- length(priors)
  posteriors <- matrix(0, n_test, K)
  
  library(matlib)
  
  p = ncol(test_data)
  result = 0
  if (cov_equal == FALSE){
    # QDA
    
    for (k in 0:9){
      # Determinant of Covariance Squared
      det_cov_sq = det(covs[,,k+1])^(-1/2)
      
      # Inverse of Covariance
      inv_cov = inv(covs[,,k+1])
      
      for (i in 1:n_test){
        mat_diff = matrix(test_data[i,]) - matrix(means[,k+1])
        posteriors[i, k+1] = (2*pi)^(-p/2) * det_cov_sq * exp((-1/2)*t(mat_diff) %*% inv_cov %*% mat_diff)
      }
    }
    
    
  } else {
    # LDA 
    
    # Determinant of Covariance Squared
    det_cov_sq = det(covs)^(-1/2)
    
    # Inverse of Covariance
    inv_cov = inv(covs)
    
    for (k in 0:9){
      for (i in 1:n_test){
        mat_diff = matrix(test_data[i,]) - matrix(means[,k+1])
        posteriors[i, k+1] = (2*pi)^(-p/2) * det_cov_sq * exp((-1/2)*t(mat_diff) %*% inv_cov %*% mat_diff)
      }
    }
  }
  
  return(posteriors)
}


Predict_labels <- function(posteriors) {
  
  #' Predict labels based on the posterior probabilities over K classes
  #' 
  #' @param posteriors A n by K posterior probabilities
  #' 
  #' @return A vector of predicted labels with length equal to n
  
  n_test <- nrow(posteriors)
  pred_labels <- rep(NA, n_test)
  
  for (i in 1:n_test){
    pred_labels[i] = which.max(posteriors[i,]) - 1
  }
  
  return(pred_labels)
}




