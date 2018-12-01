'''Libraries for Prototype selection'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import cvxpy as cvx
import math as mt
from sklearn.model_selection import KFold
import sklearn.metrics
from scipy.spatial import distance


class classifier():
    '''Contains functions for prototype selection'''
    def __init__(self, X, y, epsilon_, lambda_ ):
        '''Store data points as unique indexes, and initialize 
        the required member variables eg. epsilon, lambda, 
        interpoint distances, points in neighborhood'''
        
        self.X = X
        self.y = y
        self.epsilon = epsilon_
        self.lambda1 = lambda_
        
        self.n,self.m = self.X.shape
        self.nclass = np.max(y)+1
        self.distance = distance.cdist(X,X)
        self.neighbor = []
        for i in range(self.n):
            temp_list = []
            for j in range(self.n):
                if self.distance[i,j] < self.epsilon:
                    temp_list.append(j)
            self.neighbor.append(temp_list)
        self.alpha_opt = np.zeros((self.n,self.nclass))
        self.zeta_opt = np.zeros(self.n)
        self.A = np.zeros((self.n,self.nclass))
        self.zeta_round = np.zeros(self.n)
        self.C = np.zeros((self.n,self.nclass))
        self.result = []
        self.y_idx = []
        self.OPT = np.zeros(self.nclass)
    '''Implement modules which will be useful in the train_lp() function
    for example
    1) operations such as intersection, union etc of sets of datapoints
    2) check for feasibility for the randomized algorithm approach
    3) compute the objective value with current prototype set
    4) fill in the train_lp() module which will make 
    use of the above modules and member variables defined by you
    5) any other module that you deem fit and useful for the purpose.'''
    
    def objective_value_small(self,alpha_round,zeta_round,i):
        '''Implement a function to compute the objective value of the integer optimization
        problem after the training phase'''
        result = np.dot(self.C[:,i],alpha_round)+np.sum(zeta_round[self.y_idx[i]])  
        return result
    
    def is_feasible(self,alpha,zeta,i):
        is_feasible = True
        check1 = any(alpha < 0) or any(alpha > 1) or any(zeta[self.y_idx[i]] < 0)
        if check1:
            is_feasible = False
            return is_feasible
   
        for j in self.y_idx[i]:
            if np.sum(alpha[self.neighbor[j]]) < 1-zeta[j]:
                is_feasible = False
                return is_feasible
        return is_feasible
        
        
    def train_lp(self, verbose = False):
        '''Implement the linear programming formulation 
        and solve using cvxpy for prototype selection'''
        
        y_list = self.y.tolist()
        self.y_idx = []
        
        zeta = cvx.Variable(self.n)
        for i in range(self.nclass):
            y_idx1 = [k for k, e in enumerate(y_list) if e == i]
            y_idx2 = [k for k, e in enumerate(y_list) if e != i]
            self.y_idx.append(y_idx1)
            
            alpha = cvx.Variable(self.n)
            C = np.zeros(self.n)
            for j in range(self.n):
                C[j] = self.lambda1+len(list(set(self.neighbor[j]) & set(y_idx2)))
            self.C[:,i] = C
            
            objective = cvx.Minimize(alpha*C.reshape(self.n,1)+sum(zeta[y_idx1]))
            constraints = []
            for j in y_idx1:
                constraints.append(np.sum(alpha[self.neighbor[j]]) >= 1-zeta[j])
            constraints.append(alpha>=0)
            constraints.append(alpha<=1)
            constraints.append(zeta[y_idx1]>=0)
            prob = cvx.Problem(objective,constraints)
            result = prob.solve()
            alpha.value[alpha.value < 0] = 0
            alpha.value[alpha.value > 1] = 1
            zeta.value[zeta.value < 0] = 0
            zeta.value[zeta.value > 1] = 1
            self.alpha_opt[:,i] = alpha.value
            self.zeta_opt[y_idx1] = zeta.value[y_idx1]
            self.OPT[i] = prob.value           
        zeta_round = np.zeros(self.n)
        for i in range(self.nclass):
            alpha_round = np.zeros(self.n)
            zeta_round[self.y_idx[i]] = np.zeros(len(self.y_idx[i]))
            while(1):
                for t in range(int(2*np.log(len(self.y_idx[i])))):
                    for j in range(self.n):
                        alpha_temp = np.random.binomial(1,self.alpha_opt[j,i],1)
                        alpha_round[j] = np.maximum(alpha_round[j],alpha_temp)
                    for j in self.y_idx[i]:
                        zeta_temp = np.random.binomial(1,self.zeta_opt[j],1)
                        zeta_round[j] = np.maximum(zeta_round[j],zeta_temp)
                num1 = self.objective_value_small(alpha_round,zeta_round,i)
                num2 = 2*np.log(len(self.y_idx[i]))*self.OPT[i]
                is_feasible = self.is_feasible(alpha_round,zeta_round,i)
                if is_feasible and (num1 <= num2):
                    A_list = alpha_round.tolist()
                    self.A[:,i] = alpha_round
                    self.zeta_round[self.y_idx[i]] = zeta_round[self.y_idx[i]]
                    self.result.append([k for k, e in enumerate(A_list) if e == 1])
                    break
            
    def objective_value(self):
        '''Implement a function to compute the objective value of the integer optimization
        problem after the training phase'''
        obj = 0
        for i in range(self.nclass):
            obj += self.objective_value_small(self.A[:,i],self.zeta_round,i)
        return obj
        
    def predict(self, instances):
        '''Predicts the label for an array of instances using the framework learnt'''
        prediction = []
        cover_error = 0
        for row in instances:
            min_distance = []
            row = row.reshape(1,self.m)
            for i in range(self.nclass): 
                if self.result[i]:
                    temp_distance = distance.cdist(row,self.X[self.result[i],:],'euclidean')
                    min_distance.append(temp_distance.min())
            min_value = min(min_distance)
            if min_value > self.epsilon:
                cover_error += 1
            prediction = np.append(prediction,min_distance.index(min_value))
        return prediction, cover_error

def cross_val(data, target, epsilon_, lambda_, k, verbose):
    '''Implement a function which will perform k fold cross validation 
    for the given epsilon and lambda and returns the average test error and number of prototypes'''
    kf = KFold(n_splits=k, random_state = 42)
    score = 0
    prots = 0
    obj_val = 0
    test_error = 0
    cover_error = 0
    for train_index, test_index in kf.split(data):
        ps = classifier(data[train_index,:], target[train_index], epsilon_, lambda_)
        ps.train_lp(verbose)
        obj_val += ps.objective_value()
        predict,cover = ps.predict(data[test_index])
        score_temp = sklearn.metrics.accuracy_score(target[test_index], predict,normalize = True)
        score += score_temp
        test_error += 1-score_temp
        prots += sum(len(e) for e in ps.result)
        cover_error += cover
        
        '''implement code to count the total number of prototypes learnt and store it in prots'''
        
    score /= k
    test_error/=k
    cover_error/=k
    prots /= k
    obj_val /= k
    return score, prots, obj_val,test_error,cover_error
