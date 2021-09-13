from extras import get_X, get_Y_hat, get_sq_err_t, get_total_sq_err, avg_batch_error, err_derivative_k, err_derivative_h, get_predictions
import numpy as np 

M =35
N = 35

np.random.seed(0)
k = np.random.normal(0, 0.01, N) 

np.random.seed(1)
h = np.random.normal(0, 0.01, M)


class Feedback: 
    '''
    class for the feedback model. Parameters: 
        N = number of lags used to construct the input vector
        M = length of the feedback vector (predicted EEG)
        k = model parameter vector, stimulus
        h = model parameter vector, feedback
        lambda_k = learning rate, stimulus
        lambda_h = learning rate, feedback 
        '''
    def __init__(self, N, M, k, h, lambda_k, lambda_h): 
        self.N = N
        self.M = M
        self.k = k
        self.h = h
        self.lambda_k = lambda_k  #should people be able to change the learning rate 
        self.lambda_h = lambda_h
        
                    
    def fit(self, x, y, k, h, predictions, lambda_k = 1e-7, lambda_h = 1e-7, batchsize=500):
        '''
        Function to predict the filter coefficients using batch descent.
        
        '''
        MIN = 0
        i = int(np.floor(len(x)/batchsize))
                 
        for j in range(i): 
            
            if batchsize > x.size - j*batchsize: #if the remainder of the train set < the batchsize, the final batch = remainder 
                batchsize = x.size%(j*batchsize)
       
            MIN = j*batchsize
            T = MIN + batchsize
        
            err_deriv_k = err_derivative_k(x, y, T, predictions, k, h, MIN)
            err_deriv_h = err_derivative_h(x, y, T, predictions, k, h, MIN)
            
            k = k - lambda_k * err_deriv_k
            h = h - lambda_h * err_deriv_h
            
            predictions = get_predictions(k, h, x, T)
            
           # avg_batch_err = avg_batch_error(x, y, k, h, MIN, T, batchsize, predictions)
            #print(avg_batch_err)    
        return k, h
    
    
    'Function to predict the target signal y given the input signal x'
    def predict (self, x, M = 35, N = 35):
        Y_hat = np.zeros(M)
        size = int(x.size)
        predictions = np.zeros(size)

        for i in range(size):
            y_i =0 
            Y_hat[0]= y_i
            Y_hat = np.roll(Y_hat, 1)
            X_t = get_X(x, i, N)
            y_i = k @ X_t + h @ Y_hat 
            predictions[i] = y_i
        return predictions
        
