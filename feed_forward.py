import numpy as np

def sigmoid(z):
    a = 1.0/(1.0+np.exp(-z))
    return a

def tanh(z):
    a = (np.exp(-z) - np.exp(-z))/(np.exp(-z) + np.exp(-z))
    return a

#def softmax(y):
    

sample_size = len(t)
frequencies = f
input = Sxx
for i in range(120,121):
    xi = Sxx[:,0:10]
    #print xi.shape
    input_gate = sigmoid(np.dot(np.transpose(w_xi), xi)
                         + (w_hi * hidden_state_prev) 
                         + (w_ci * cell_state_prev) + 1)
    forget_gate = sigmoid(np.dot(np.transpose(w_xf), xi) + 
                          (w_hf * hidden_state_prev) + 
                          (w_cf * cell_state_prev) + 1)
    cell_state_gate = (forget_gate * cell_state_prev) + (input_gate * tanh(np.dot(
        np.transpose(w_xc), xi) + (w_hc * hidden_state_prev) + 1))
    output_gate = sigmoid(np.dot(np.transpose(w_xo), xi) 
                     + (w_ho * hidden_state_prev) 
                     + (w_co * cell_state_gate) + 1)
    hidden_state = output_gate * tanh(cell_state_gate)
    y = np.dot(np.transpose(w_hy), hidden_state) + 1
    
    

#print input_gate
#print forgot_gate
#print cell_state_gate
#print output_gate
#print hidden_state
#print tanh(cell_state_gate)
#print y
#print tanh(np.dot(
#        np.transpose(w_xc), xi) + (w_hc * hidden_state_prev) + 1)