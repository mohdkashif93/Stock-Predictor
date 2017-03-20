from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
def ridge_reg():
    model = linear_model.Ridge()
    params = {'alpha':[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]}
    return model,params


def randomForest_reg():
	clf = RandomForestRegressor(random_state=0)
	params = {'n_estimators':list(xrange(10,100,10)),'max_features' : ['auto','log2','sqrt']}
	return clf,params


def lasso_reg():
	clf = linear_model.Lasso()
	params = {'alpha':[1e-6,1e-5,1e-4,1e-3,1e-2, 1e-1, 1, 10, 100, 1000, 10000]}
	return clf,params

def ann_reg():
	clf = MLPRegressor()
	params = {'activation' : ['identity', 'logistic', 'tanh', 'relu'],
#			'solver' : ['sgd', 'adam'],
			'alpha':[1e-6,1e-5,1e-4,1e-3,1e-2, 1e-1, 1, 10],
			'learning_rate':['constant','invscaling','adaptive'],
			'learning_rate_init':[1e-3,1e-2,1e-1,1,10]
			}
	return clf,params


