#parts of the code are taken from two tutorial,namels https://nathanrooy.github.io/posts/2017-08-27/simple-differential-evolution-with-python/ and
#https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/#


import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """
    Function to minimize. (Levy1D see https://www.sfu.ca/~ssurjano/levy.html). Global min value: 0.0
    """
    w0 = (1 + (x[0] - 1) / 4)
    term1 = np.power(np.sin(np.pi * w0), 2)

    term2 = 0
    for i in range(len(x) - 1):
        wi = 1 + (x[i] - 1) / 4
        term2 += np.power(wi - 1, 2) * (1 + 10 * np.power(np.sin(wi * np.pi + 1), 2))

    wd = (1 + (x[-1] - 1) / 4)
    term3 = np.power(wd - 1, 2)
    term3 *= (1 + np.power(np.sin(2 * np.pi * wd), 2))

    y = term1 + term2 + term3
    return y

best_values = []
corresponding_function_values = []
class DEOptimizer():
    """
    DE Optimizer
    :param max_iter: max number of iterations
    :param D: dimension of the problem being solved
    :param f: function to be optimized
    :param NP: population size
    :param F: scaling factor
    :param CR: crossover rate
    :return: the best individual with the best function value
    """
    np.random.seed(2)

    def __init__(self,bounds,Np):

        # TODO Initialize DE parameters and random population
        self.pop = []
        self.bounds = bounds
        self.dimensions= len(bounds)
        self.pop = np.random.rand(Np, self.dimensions)
        print('Initalization of random pouplation completed')
        
        
    def ensure_bounds(self,vec, bounds):
        vec_new = []
        # cycle through each variable in vector 
        for i in range(len(vec)):
            # variable exceedes the minimum boundary
            if vec[i] < bounds[i][0]:
                vec_new.append(bounds[i][0])
            # variable exceedes the maximum boundary
            if vec[i] > bounds[i][1]:
                vec_new.append(bounds[i][1])
            # the variable is fine
            if bounds[i][0] <= vec[i] <= bounds[i][1]:
                vec_new.append(vec[i])
        return vec_new

    # TODO implement mutation operation
    def mutation(self,curr_index,random_inidividuals,pouplation_size,mut_factor):
        v = []
        idxs = [idx for idx in range(pouplation_size) if idx != curr_index]
        a, b, c = random_inidividuals[np.random.choice(idxs, 3, replace = False)]
        v=np.clip(a + mut_factor * (b - c), 0, 1)
        return v

    # TODO implement crossover operation
    def crossover(self,mutant,crossp,min_b,diff,fitness,fobj,j,best_idx,best):
        cross_points = np.random.rand(self.dimensions) < crossp
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        trial = np.where(cross_points, mutant, self.pop[j])
        trial_denorm = min_b + trial * diff
        f = fobj([trial_denorm])
        if f < fitness[j]:
            fitness[j] = f
            self.pop[j] = trial
            if f < fitness[best_idx]:
                best_idx = j
                best = trial_denorm
        return best,best_idx

    # TODO implement DE optimization loop
    def DE_optimization_loop(self,fobj, bounds, mut=0.5, crossp=0.5, Np=10, its=1000):
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + self.pop * diff
        fitness = np.asarray([fobj([ind]) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        for i in range(its):
            for j in range(Np):
                mutant = self.mutation(j,self.pop,Np,mut)
                mutant= self.ensure_bounds(mutant,bounds)
                best, best_idx = self.crossover(mutant,crossp,min_b,diff,fitness,fobj,j,best_idx,best)
                best_function_value = fitness[best_idx]
            best_values.append(best[0])
            corresponding_function_values.append(best_function_value[0])
        return best_values,corresponding_function_values


bounds =[(-15, 10)]#bound
its = 200#maximum iterations 
f_value=0.5,#mutation f value
crossp=0.5,#crossover parameter
Np=10#pouplation size
DE = DEOptimizer(bounds=[(-15, 10)],Np=10)
best = DE.DE_optimization_loop(lambda x: f(x),bounds,f_value,crossp,Np,its)
print('the best individul:',best[0][-1])
print('the best function value:',best[1][-1])


plt.plot(best_values, corresponding_function_values, marker='^', label='best data_points vs their function_values')
plt.title('per iteration best data point and their corresponding function value')
plt.xlabel('points selected from the pouplation')
plt.ylabel('Function value(evaluation) of corresponding data points')
plt.legend()
plt.savefig('scatter.png')
plt.show()
