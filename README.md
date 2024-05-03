# kan-ode
Rough experiments on using Kolmogorov-Arnold Networks (KANs) as the replacement layers in a latent dynamics function, e.g. as the layers of a neural ODE.

### Data:
We'll consider two types of simple 1-dimensional ODE functions - influenced by 
either an additive control or a multiplicative control. The gray zones representing interpolation/training bounds while
blue and red zones represent extrapolation bounds. While we could consider bound ranges that exhibit fast and slow changes
to the resulting trajectory, for now we'll just consider the fast bounds and interpolation.

We consider a fixed initial condition such that the only influence exerted is the control.
The model is only given x0 and c and is asked to forecast the remaining N timesteps forward using its dynamics function f_theta, 
represented by a neural ordinary differential equation. In this case, we are testing whether replacing the neural ODE's layers with KAN Layers
can result in system identification properties given its sparsity and symbolic regression patterns. 
![img.png](figures/img.png)

Later potential systems to consider will be higher dimensional Hamiltonian systems like Bouncing Ball, Pendulum, Lotka-Volterra, etc etc.

### Notes:
Honestly unsure how much time/effort I'll put into this repo but happy for any discussions on the matter.
Primarily interested to see if, in a latent setting with high-dimensional observations, KANs can infer the actual equations
underlying the system in an interpretable way/an equation via the symbolic regression component and sparsity regularization.

If you're unfamiliar with the realm of neural SSMs or sequential latent variable models, here are some resources:
- https://proceedings.neurips.cc/paper_files/paper/2021/file/54b2b21af94108d83c2a909d5b0a6a50-Paper.pdf
- https://arxiv.org/abs/2111.05458v1
- https://openreview.net/pdf?id=7C9aRX2nBf2
- https://github.com/qu-gg/torch-neural-ssm (self-plug, but hey, it is an overview)


### Results:
Thus far in experiments I've noticed the necessity of using a linear transformation into a latent space (e.g. 2D data space -> 4D latent space), 
otherwise the method has an incredibly high initial loss and won't optimize well under Adam.

I haven't done much hyperparameter or initialization tuning so results may not be super meaningful yet.

Here is an MLP converged (black GT, blue predictions):

![reconstructedControls.png](experiments%2F125125125_multiplicative_additive%2Fadditive%2Fversion_0%2Ftest%2FreconstructedControls.png)

And here is a data-space KAN:

![reconstructedControls.png](experiments%2F125125125_multiplicative_kan_additive%2Fkan_additive%2Fversion_0%2Ftest%2FreconstructedControls.png)

I'm running a latent-space KAN currently and will post the results when it is done.