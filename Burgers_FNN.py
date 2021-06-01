from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot
import deepxde as dde

'''
In DeepXDE, the coding sequence is:
1. Define the DE system involving the Differential equation, BC\IC, Geometry(domain), Training data(or choices of residue points).
2. Building the model with constructed DE system by choosing appropriate network structure.
3. Using the model to train.
'''
def func(x):
    if x >= 0:
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
     For the choice of the residue points, we have the following options:
     "uniform” (equispaced grid), “pseudo” (pseudorandom), “LHS” (Latin hypercube sampling), “Halton” (Halton sequence), “Hammersley” (Hammersley sequence), or “Sobol” (Sobol sequence).
     '''
    
    def exact_sol(x):
        return 1 - np.heaviside(x[:, 0:1] - 0.5*x[:, 1:2], 1)

    
    data = dde.data.TimePDE(
        geomtime, pde, [ic], num_domain=2540, num_boundary=80, num_initial=160, solution=exact_sol, num_test=1000
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
        x_interval = 100
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
    np.savetxt("test_FNN.dat", np.hstack((X, y_true, y_pred)))
    
    data = np.genfromtxt("test_FNN.dat", delimiter=' ')
    x = data[:, 0]
    t = data[:, 1]
    u_exact = data[:, 2]
    u_predict = data[:, 3] 
    fig, ax = pyplot.subplots(nrows=3)
    
    ax[0].tricontour(x, t, u_predict, levels=14, linewidths=0.5, colors='k')
    cntr_0 = ax[0].tricontourf(x, t, u_predict, levels=14, cmap="RdBu_r")
    ax[0].plot(x, t, 'k.', ms=3)
    fig.colorbar(cntr_0, ax=ax[0])
    
    ax[1].tricontour(x, t, u_exact, levels=14, linewidths=0.5, colors='k')
    cntr_1 = ax[1].tricontourf(x, t, u_exact, levels=14, cmap="RdBu_r")
    ax[1].plot(x, t, 'k.', ms=3)
    fig.colorbar(cntr_1, ax=ax[1])
    
    ax[2].tricontour(x, t, u_predict-u_exact, levels=14, linewidths=0.5, colors='k')
    cntr_2 = ax[2].tricontourf(x, t, u_predict-u_exact, levels=14, cmap="RdBu_r")
    ax[2].plot(x, t, 'k.', ms=3)
    fig.colorbar(cntr_2, ax=ax[2])
    
    pyplot.show()
    
    x_last = x[9900:10000]
    u_exact_last = u_exact[9900:10000]
    u_predict_last = u_predict[9900:10000]
    pyplot.xlabel("x")
    pyplot.ylabel("u")
    pyplot.plot(x_last, u_exact_last, color="red", linestyle="dashed", label="u_exact")
    pyplot.plot(x_last, u_predict_last, color="orange", linestyle="dashed", label="u_predict")
    pyplot.legend()
    pyplot.show()

   

    
    

if __name__ == "__main__":
    main()
