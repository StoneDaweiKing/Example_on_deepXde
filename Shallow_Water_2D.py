from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde

def main():
    '''
    Here y = [h u v]^T, x = [x y t]^T    
     '''
    def pde(x, y):
        print(x.shape)
        print(y.shape)
        h=y[:,0:1]
        u=y[:,1:2]
        v=y[:,2:3]
        dh_x = dde.grad.jacobian(y, x, i=0, j=0)
        dh_y = dde.grad.jacobian(y, x, i=0, j=1)
        dh_t = dde.grad.jacobian(y, x, i=0, j=2)
        du_x = dde.grad.jacobian(y, x, i=1, j=0)
        du_y = dde.grad.jacobian(y, x, i=1, j=1)
        du_t = dde.grad.jacobian(y, x, i=1, j=2)
        dv_x = dde.grad.jacobian(y, x, i=2, j=0)
        dv_y = dde.grad.jacobian(y, x, i=2, j=1)
        dv_t = dde.grad.jacobian(y, x, i=2, j=2)
        
        first = (
            dh_t + dh_x*u + du_x*h + dh_y*u + du_y*h
        )
        second = (
            dh_t*u + du_t*h + (dh_x*u*u + 2*du_x*u*h + h*dh_x) + (dh_y*u*v + du_y*h*v + dv_y*h*u)
        )
        third = (
            dh_t*v + dv_t*h + (dh_y*v*v + 2*dv_y*v*h + h*dh_y) + (dh_x*u*v + du_x*h*v + dv_x*h*u)
        )
        return [first, second, third]
    
    '''
     Define the domain of the DE system.
     '''
    geom = dde.geometry.Rectangle(xmin=[-1, -1], xmax=[1, 1])
    timedomain = dde.geometry.TimeDomain(0, 3)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    '''
     Define the initial condition of the DE system.
     '''        
        
    ic_h = dde.IC(
        geomtime, lambda x: 1 + (np.heaviside(x[:, 0:1] + 0.5, 1) - np.heaviside(x[:, 0:1] - 0.5, 1)) * (np.heaviside(x[:, 1:2] + 0.5, 1) - np.heaviside(x[:, 1:2] - 0.5, 1)), lambda _, on_initial: on_initial, component = 0
    )
    ic_u = dde.IC(
        geomtime, lambda x: 0, lambda _, on_initial: on_initial, component = 1
    )
    ic_v = dde.IC(
        geomtime, lambda x: 0, lambda _, on_initial: on_initial, component = 2
    )
    
    data = dde.data.TimePDE(
        geomtime, pde, [ic_h, ic_u, ic_v], num_domain=2540, num_boundary=80, num_initial=160
    )
    
    '''
     Build the model with 3 hidden layers, in each layers we have the width being 20
     '''
    net = dde.maps.FNN([3] + [20] * 3 + [3], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    
    '''
     Train the model within 15000 iterations and learning rate being 1*10-3.
     '''
    model.compile("adam", lr=1e-3)
    model.train(epochs=8000)
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()
    
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
if __name__ == "__main__":
    main()