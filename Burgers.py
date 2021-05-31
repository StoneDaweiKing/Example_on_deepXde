from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde

'''
In DeepXDE, the coding sequence is:
1. Define the DE system involving the Differential equation, BC\IC, Geometry(domain), Training data(or choices of residue points).
2. Building the model with constructed DE system by choosing appropriate network structure.
3. Using the model to train.
'''
def func(x):
    if x > 0:
        return 0
    else:
        return 1


def main():
    '''
     In DeepXDE, we would call deepxde.grad module to define the deriavative (which is through the tensorflow module)
     We would use grad.jacobian to represent first deriavative; grad.hessian to represent second deriavative
     In the code below, x represents the input (which is a two-column matrix with the first column represents x and
     second column represents y). y represents the output (which is a one-column matrix representing u(x, t)).
     '''
    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        return dy_t + y * dy_x
    
    '''
     Define the domain of the DE system.
     '''
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    '''
     Define the initial condition of the DE system.
     '''
    ic = dde.IC(
        geomtime, lambda x: 1 - np.heaviside(x[:, 0:1], 1), lambda _, on_initial: on_initial
    )

    '''
     Choosing the residue points. Here we choose 2540 residue points in the whole domain, and 80 residue points on the boundary, 160 points
     for the initial condition.
     '''
    data = dde.data.TimePDE(
        geomtime, pde, [ic], num_domain=2540, num_boundary=80, num_initial=160
    )
    
    '''
     Build the model with 3 hidden layers, in each layers we have the width being 20
     '''
    net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    
    '''
     Train the model within 15000 iterations and learning rate being 1*10-3.
     '''
    model.compile("adam", lr=1e-3)
    model.train(epochs=15000)
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()
    
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    '''
     The following part has nothing related to the deepXDE itself. It's just for ploting
     relative loss (the loss between the exact solution and predicted solution) of DE.
     Sorry that I didn't find the appropriate API for making comparison to exact solution.
     So I write some dummy code to do this part.
     '''
    def gen_testdata():
        x_interval = 500
        t_interval = 100
        X = np.zeros((x_interval * t_interval, 2))
        y = np.zeros((x_interval * t_interval, 1))
        for i in range(0, t_interval):
            for j in range(0, x_interval):
                t = i / x_interval
                x = -1 + j * 2 / x_interval
                X[i * t_interval + j, 0] = x
                X[i * t_interval + j, 1] = t
                if x >= 0.5*t:
                    y[i * t_interval + j, 0] = 0
                else:
                    y[i * t_interval + j, 0] = 1
        return X, y
    
    X, y_true = gen_testdata()
    y_pred = model.predict(X)
    f = model.predict(X, operator=pde)
    print("Mean residual:", np.mean(np.absolute(f)))
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
                    
    

if __name__ == "__main__":
    main()
