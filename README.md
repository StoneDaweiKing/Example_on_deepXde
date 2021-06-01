# Burgers Equation
For Burgers equation <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;u&space;}{\partial&space;t}&space;&plus;&space;\frac{\partial&space;u^2}{\partial&space;x}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/png.latex?\frac{\partial&space;u&space;}{\partial&space;t}&space;&plus;&space;\frac{\partial&space;u^2}{\partial&space;x}&space;=&space;0" title="\frac{\partial u }{\partial t} + \frac{\partial u^2}{\partial x} = 0" /></a> with boundary condition <a href="https://www.codecogs.com/eqnedit.php?latex=u(x,&space;0)&space;=&space;\left\{\begin{matrix}0&space;\&space;\&space;\text{if&space;x&space;$\ge$&space;0}&space;\\1&space;\&space;\&space;\text{if&space;x&space;$<$&space;0}&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/png.latex?u(x,&space;0)&space;=&space;\left\{\begin{matrix}0&space;\&space;\&space;\text{if&space;x&space;$\ge$&space;0}&space;\\1&space;\&space;\&space;\text{if&space;x&space;$<$&space;0}&space;\end{matrix}\right." title="u(x, 0) = \left\{\begin{matrix}0 \ \ \text{if x $\ge$ 0} \\1 \ \ \text{if x $<$ 0} \end{matrix}\right." /></a>.
We would use PINN to solve the equation:
## FNN (Feedforward Neural Network) with 15000 adams + L-BFGS-B
Mean residual: 0.12544216, L2 relative error: 0.1371976435
Contour Plot: The first graph is for predicted, the second graph is for exact solution and the third graph is for residual.
<img src=img\Contour%20Plot%20FNN.png>

Train Loss and Test Loss (Here the test loss is calculated upon the 10000 uniform points on x-t plane).
<img src=img\FNN.png>

3D plot for the predicted value:
(Over 10000 uniform residul points in the x-t domain)

<img src=img\FNN_3d.png>

The comparison plot between expected value and exact value:
<img src=img\FNN_t%3D1.png>
