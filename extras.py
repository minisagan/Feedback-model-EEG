import numpy as np

M = 35
N = 35

np.random.seed(0)
k = np.random.normal(0, 0.01, N) 

np.random.seed(1)
h = np.random.normal(0, 0.01, M)
 
'''----------------------------------------------------------------------------------------------------------------------------------------
functions to get X_t (lagged stimulus input vector) and Y^_t-1 (our EEG predictions)
data = np.hstack([data[f'stim/part{0}'][:]])
t = the specific time
N = length of X_t
----------------------------------------------------------------------------------------------------------------------------------------'''

def get_X(x, t, N): #t is the specific time and N is the legnth of the vector
    X = np.zeros(N)
    if t >= N: 
        X = np.flip(x[t-N+1:t+1]) 
    if t < N:
        X[:t] = np.flip(x[:t]) 
        
    return X


def get_predictions(k, h, x, stop):
    Y_hat = np.zeros(M)
    predictions = np.zeros(stop)

    for i in range(stop):
        y_i =0 
        Y_hat = np.roll(Y_hat, 1)
        Y_hat[0]= y_i
        X_t = get_X(x, i, N)
        y_i = k @ X_t + h @ Y_hat 
        predictions[i] = y_i
    return predictions




def get_Y_hat(t, predictions):
    Y_hat = np.zeros(M)
    if t>= M: 
        Y_hat = np.flip(predictions[t-M : t])
    if t<M: 
        Y_hat[:t] =  np.flip(predictions[ : t])   
    return Y_hat

'''----------------------------------------------------------------------------------------------------------------------------------------
functions to get the errors 
stop - start = the times we want to sum errors between, for our first batch start = 0, stop = batchsize
----------------------------------------------------------------------------------------------------------------------------------------'''

def get_sq_err_t(x, y, k, h, s, t, predictions): 
    Y_hat = get_Y_hat(t, predictions) #note that this gives t-1 since ranges are not inclusive
    X = get_X(x, t, N)
    y_hat = k @ X + h @ Y_hat
    squared_error = (y_hat - y[t])**2
    return squared_error 


def get_total_sq_err(x, y, k, h, start, stop, predictions):
    total_errors = []
    for t in range(start, stop+1): 
        squared_error = get_sq_err_t(x, y, k, h, start, t, predictions)
        total_errors.append(squared_error)
    total_sq_err = np.sum(total_errors)
    return total_sq_err   


def avg_batch_error(x, y, k, h, start, stop, batchsize, predictions): 
    tot_error= get_total_sq_err(x, y, k, h, start, stop, predictions)
    avg_batch_err = tot_error/batchsize
    return(avg_batch_err)

'''----------------------------------------------------------------------------------------------------------------------------------------
functions to get the derivatives of the errors for all times up until t
k & h  = parameters
----------------------------------------------------------------------------------------------------------------------------------------'''

def err_derivative_k(x, y, t, predictions, k=k, h=h, start = 0):
    derivative_array_k = np.zeros((M,N)) #array of zeroes to fill with the derivatives (d/dk(yhat_t-1),...,d/dk(yhat_t-M))
    Y_hat = np.zeros(M) #array of zeroes to fill with the values of yhat_t-1,...,yhat_t-M
    grad_times_error_array_k = np.zeros((t-start,N))
    for i in range(start, t): 
        X = get_X(x, i, N)
        
        derivative_array_k = np.roll(derivative_array_k, 1, axis =0) #shifts all elements of the first row down one, and makes first row contain 0s
        d_dk = X + h @ derivative_array_k #formula for the derivative of y_hat at time t, basically eliminitates k this will be a vector
        derivative_array_k[0,:] = d_dk.T #replaces first row with derivatives 
        
        y_t = y[i] #actual eeg value at time t, Y_t = (y_0,...,y_t).T SHOULD THIS BE t-1? 
        Y_hat = get_Y_hat(i, predictions) 
        if Y_hat.size != 35: 
            continue
        y_hat = k @ X + h @ Y_hat 
        #print(y_hat)
    
        grad_times_error_k = d_dk * (y_hat - y_t)
        grad_times_error_array_k[i-start-1,:] = grad_times_error_k.T
    error_deriv_wrt_k = 2 * np.sum(grad_times_error_array_k, axis = 0)
        
    return error_deriv_wrt_k


def err_derivative_h(x, y, t, predictions, k=k, h=h, start = 0): 
    derivative_array_h = np.zeros((M,M))
    Y_hat = np.zeros(M)
    grad_times_error_array_h = np.zeros((t-start,M))

    for i in range(start, t):
        Y_hat = get_Y_hat(i, predictions)
        derivative_array_h = np.roll(derivative_array_h, 1, axis = 0)
        if Y_hat.size != 35: 
            continue
        d_dh = Y_hat + h @ derivative_array_h
    
        derivative_array_h[0,:] = d_dh.T
    
        
        X = get_X(x, i, N)
        y_t = y[i]
        y_hat = k @ X + h @ Y_hat
        
        grad_times_error_h = d_dh * (y_hat - y_t)
        grad_times_error_array_h[i-start-1,:] = grad_times_error_h.T
    
    error_deriv_wrt_h = 2 * np.sum(grad_times_error_array_h, axis = 0)
    return error_deriv_wrt_h

#err_derivative_h(40,k,h, start = 0)




