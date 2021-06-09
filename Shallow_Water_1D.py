from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot
import deepxde as dde
import os
import imageio

def main():
    '''
    Here y = [h u v]^T, x = [x t]^T    
     '''
    def pde(x, y):
        h=y[:,0:1]
        u=y[:,1:2]
        v=y[:,2:3]
        dh_x = dde.grad.jacobian(y, x, i=0, j=0)
        dh_t = dde.grad.jacobian(y, x, i=0, j=1)
        du_x = dde.grad.jacobian(y, x, i=1, j=0)
        du_t = dde.grad.jacobian(y, x, i=1, j=1)
        dv_x = dde.grad.jacobian(y, x, i=2, j=0)
        dv_t = dde.grad.jacobian(y, x, i=2, j=1)
        first = (
            dh_t + dh_x*u + du_x*h
        )
        second = (
            dh_t*u + du_t*h + (dh_x*u*u + 2*du_x*u*h + h*dh_x)
        )
        third = (
            dh_t*v + dv_t*h + (dh_x*u*v + h*du_x*v + h*u*dv_x)
        )
        return [first, second, third]
    
    '''
     Define the domain of the DE system.
     '''
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 0.51)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    '''
     Define the initial condition of the DE system.
     '''        
        
    ic_h = dde.IC(
        geomtime, lambda x: 2 - np.heaviside(x[:, 0:1], 1), lambda _, on_initial: on_initial, component = 0
    )
    ic_u = dde.IC(
        geomtime, lambda x: 0, lambda _, on_initial: on_initial, component = 1
    )
    ic_v = dde.IC(
        geomtime, lambda x: 0, lambda _, on_initial: on_initial, component = 2
    )
    
    data = dde.data.TimePDE(
        geomtime, pde, [ic_h, ic_u, ic_v], num_domain=2540, num_boundary=80, num_initial=160, num_test = 10000
    )
    
    '''
     Build the model with 3 hidden layers, in each layers we have the width being 20
     '''
    net = dde.maps.FNN([2] + [20] * 3 + [3], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    
    '''
     Train the model within 10000 iterations and learning rate being 1*10-3.
     '''
    model.compile("adam", lr=1e-3)
    model.train(epochs=10000)
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
    
    filenames = []
    times = [];
    
    
    for file in os.listdir("ShallowData_1d"):
        data = np.loadtxt(os.path.join("ShallowData_1d", file), delimiter='\t', skiprows=1, dtype=float);
        time = data[1, 4]
        times.append(time)
        X = data[:,3:5]
        y_pred = model.predict(X);
        fig, ax = pyplot.subplots(nrows=3);
        ax[0].plot(X[:, 0], y_pred[:, 0], label="predict");
        ax[0].plot(X[:, 0], data[:, 0], label="exact");
        ax[0].set_title(f"h-x at time {time}")
        ax[1].plot(X[:, 0], y_pred[:, 1], label="predict");
        ax[1].plot(X[:, 0], data[:, 1], label="exact");
        ax[1].set_title(f"u-x at time {time}")
        ax[2].plot(X[:, 0], y_pred[:, 2], label="predict");
        ax[2].plot(X[:, 0], data[:, 2], label="exact");
        ax[2].set_title(f"v-x at time {time}")
        pyplot.savefig(f'{time}.png')
        filenames.append(f'{time}.png')
        pyplot.close()

    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
                         
    for filename in set(filenames):
        os.remove(filename)
    
    
    
    def gen_testdata(t):
        x_interval = 10000;
        X = np.zeros((x_interval, 2));
        for i in range(0, x_interval):
            X[i, 0] = -1 + 2 * i / x_interval
            X[i, 1] = t
        y_pred = model.predict(X);
        fig, ax = pyplot.subplots(nrows=3);
        ax[0].plot(X[:, 0], y_pred[:, 0]);
        ax[0].set_title(f"h-x at time {t}")
        ax[1].plot(X[:, 0], y_pred[:, 1]);
        ax[1].set_title(f"u-x at time {t}")
        ax[2].plot(X[:, 0], y_pred[:, 2]);
        ax[2].set_title(f"v-x at time {t}")
        pyplot.show();
        print('I am here to plot!')
        
    gen_testdata(0);
    gen_testdata(0.25);
    gen_testdata(0.5);
    
    
    
    

    


    
if __name__ == "__main__":
    main()