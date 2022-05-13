import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import os 
path = os.getcwd()
parent = os.path.join(path, os.pardir)
sys.path.append(os.path.abspath(parent))
from bayes_opt1 import BayesianOptimization
from bayes_opt1 import UtilityFunction
sys.path.append(os.path.abspath(os.path.join(parent, 'bayes_opt1')))
from util import _kg2



def normalization(utility):
    return (utility- min(utility)) / (max(utility) - min(utility))


def plot_gp(optimizers, x, target, params, n_init=2): 

    x = x.reshape(-1,1)
    y = target(x)
    
    def posterior(optimizer, x_obs, y_obs, grid):
        optimizer._gp.fit(x_obs, y_obs)
        mu, sigma = optimizer._gp.predict(grid, return_std=True)
        return mu, sigma
    
    fig = plt.figure(figsize=(16, 10))
    steps = len(list(optimizers.values())[0].space)
  
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 2]) 
    axis = plt.subplot(gs[0])
    af = plt.subplot(gs[1])
    n = len(optimizers)
    colmap = plt.cm.get_cmap(name='rainbow', lut=n)

    
    x_obs = {} 
    y_obs={}
    mu={}
    sigma={}
    utility={}
    axis.plot(x, y, linewidth=3, label='Target')
    
    i=0
    for acq, optimizer in optimizers.items(): 
        x_obs_i = np.array([[res["params"]["x"]] for res in optimizer.res])
        x_obs[acq] = x_obs_i
        y_obs_i = np.array([res["target"] for res in optimizer.res])
        y_obs[acq] =  y_obs_i
        mu[acq], sigma[acq] = posterior(optimizer, x_obs_i, y_obs_i, x)
        utility_function = UtilityFunction(kind=acq, kappa = params[acq]['kappa'], xi = params[acq]['xi'])
        if acq != 'kg':
            utility[acq] = utility_function.utility(x, optimizer, optimizer._gp, optimizer._space.target.max())
        else:
            utility[acq] = _kg2(x, optimizer, optimizer._gp, n_grid = 100, J = 30)
        axis.plot(x, mu[acq], '--', color=colmap(i), label='Prediction')
        i=i+1
    
    fig.suptitle(
        'Utility Functions After {} Iterations And {} Initial Points'.format(steps-n_init, n_init),
        fontdict={'size':50}
    )
    
    axis.set_xlim((min(x), max(x)))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
 

    for i, acq in enumerate(utility):
        af.plot(x, normalization(utility[acq]), label=acq, color= colmap(i))
        af.plot(x[np.argmax(utility[acq])], np.max(normalization(utility[acq])), '*', markersize=15, markerfacecolor=colmap(i), markeredgecolor='k', markeredgewidth=1)


    af.set_xlim((min(x), max(x)))
    af.set_ylim((-0.2,1.2))
    af.set_ylabel('Utility', fontdict={'size':20})
    af.set_xlabel('x', fontdict={'size':20})
    

    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    af.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    plt.show()
    
    
    
def plot_convergence(optimizers, x, target, params, it=20, init_point=2):
    x = x.reshape(-1,1)
    y = target(x)
    
    tar={}
    points={}

    utility_function={}
    for acq, optimizer in optimizers.items(): 
        utility_function[acq] = UtilityFunction(kind=acq, kappa = params[acq]['kappa'], xi = params[acq]['xi'])
        points[acq]=np.zeros(it)
        tar[acq]= np.zeros(it)
        for i in range(it):    
            points[acq][i]=optimizer.suggest(utility_function[acq])['x']
            tar[acq][i]= target(points[acq][i])
            optimizer.register(params=points[acq][i], target=tar[acq][i])
            

    
    fig = plt.figure(figsize=(13, 6))
    
    af = plt.subplot()
    
    fig.suptitle(
        'Convergences To The Optimum After {} Iterations And {} Initial Points'.format(it, init_point),
        fontdict={'size':50}
    )
    
    
    steps = len(list(optimizers.values())[0].space)
    n = len(optimizers)
    colmap = plt.cm.get_cmap(name='rainbow', lut=n)
    
    num_iter=np.arange(1,it+1)
    
    for i, acq in enumerate(optimizers.keys()):
        af.plot(num_iter, points[acq], '*',markersize=15,markerfacecolor=colmap(i), markeredgecolor='k', markeredgewidth=1,label=acq,linestyle='solid',color=colmap(i))
    
 
    af.axhline(y=x[np.argmax(target(x))], linestyle=':', label='Optimum to be achieved')
    af.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
   

    a = range(0,21)
    af.set_xticks(a)
    af.set_yticks(range(-2,11))
    plt.xlabel('Number of iterations')
    plt.ylabel('Suggested x')
    
    
    plt.show()


# Plot regret

def simple_regret(f_max, target_space):
    """
    The simple regret rT = max{x∈X} f(x) − max{t∈[1,T]} f(x_t) measures the value of
the best queried point so far. 
    """
    return f_max-target_space.max()

def plot_simple_regret(optimizers, x, target, params, dim, it=20, init_point=2):
    
    tar={}
    points={}
    utility_function={}
    regrets={}
    
    if dim > 1:
        inputs = []
        for col in range(dim):
            inputs.append(x[:,col])
        y = target(*inputs)


        f_max=max(y)

        for acq, optimizer in optimizers.items():  
            regrets[acq]=np.zeros(it)
            utility_function[acq] = UtilityFunction(kind=acq, kappa = params[acq]['kappa'], xi = params[acq]['xi'])

        for i in range(it):
            for acq, optimizer in optimizers.items():
                points[acq]=optimizer.suggest(utility_function[acq])
                tar[acq]= target(**points[acq])
                optimizer.register(params=points[acq], target=tar[acq])
                regrets[acq][i]= simple_regret(f_max, optimizer._space.target)
    else:
        x = x.reshape(-1,1)
        y = target(x)
        
        f_max=max(y)

        for acq, optimizer in optimizers.items(): 
            points[acq]=np.zeros(it)
            tar[acq]= np.zeros(it)
            regrets[acq]=np.zeros(it)
            utility_function[acq] = UtilityFunction(kind=acq, kappa = params[acq]['kappa'], xi = params[acq]['xi'])

        for i in range(it):
            for acq, optimizer in optimizers.items():
                points[acq][i]=optimizer.suggest(utility_function[acq])['x']
                tar[acq][i]= target(points[acq][i])
                optimizer.register(params=points[acq][i], target=tar[acq][i])
                regrets[acq][i]= simple_regret(f_max, optimizer._space.target)

    
    fig = plt.figure(figsize=(13, 6))
    
    af = plt.subplot()
    
    fig.suptitle(
        'Simple Regret After {} Iterations And {} Initial Points'.format(it, init_point),
        fontdict={'size':50}
    )
    
    steps = len(list(optimizers.values())[0].space)
    n = len(optimizers)
    colmap = plt.cm.get_cmap(name='rainbow', lut=n)
    
    num_iter=np.arange(1,it+1)
    
    for i, acq in enumerate(optimizers.keys()):
        af.plot(num_iter, regrets[acq], '*',markersize=15,markerfacecolor=colmap(i), markeredgecolor='k', markeredgewidth=1,label=acq,linestyle='solid',color=colmap(i))


    af.set_ylim((-0.2,0.4))
    af.axhline(y=0, linestyle=':', label='0.0')
    af.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    
    
    af.set_xticks(num_iter)
    
    plt.xlabel('Number of iterations')
    plt.ylabel('Simple regret')
    
    
    plt.show()
    
    
    
