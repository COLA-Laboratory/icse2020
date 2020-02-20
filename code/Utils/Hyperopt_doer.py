from Algorithms.Framework import cpdp
from Algorithms.Framework import *
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

class optParamAll(object):
    def __init__(self, sx, sy, tx, ty, loc, classifier, adaptation, fe=1000):
        self.sx = sx
        self.sy = sy
        self.tx = tx
        self.ty = ty
        self.loc = loc
        self.adaptation = adaptation
        self.clf = classifier
        self.fe = fe
        self.trails = Trials()

    def objFunc(self, params):
        self.p = cpdp(clf=self.clf, adpt=self.adaptation)
        self.p.set_params(**params)
        sx = self.sx
        sy = self.sy
        tx = self.tx
        ty = self.ty
        loc = self.loc
        res = self.p.run(sx, sy, tx, ty, loc)

        return {'loss': -np.mean(res), 'status': STATUS_OK, 'result': np.mean(res)}

    def run(self):
        trails = Trials()

        if self.adaptation == 'TCA':
            adptdefault_value = {
                'kernel_type': 'linear',
                'dim': 5,
                'lamb': 1,
                'gamma': 1
            }
            adptparamSpace = {
                'kernel_type': hp.choice('kernel_type', ['primal', 'linear', 'rbf', 'sam']),
                'dim': hp.choice('dim', range(5, max(self.sx.shape[1], self.tx.shape[1]))),
                'lamb': hp.uniform('lamb', 1e-6, 1e2),
                'gamma': hp.uniform('gamma', 1e-5, 1e2)
            }

        if self.adaptation == 'DBSCANfilter':
            adptdefault_value = {
                'eps': 1,
                'min_samples': 10
            }
            adptparamSpace = {
                'eps': hp.uniform('eps', 0.1, 1e2),
                'min_samples': hp.choice('min_samples', range(1, 100))
            }

        if self.adaptation == 'Bruakfilter':
            adptdefault_value = {
                'n_neighbors': 10
            }
            adptparamSpace = {
                'n_neighbors': hp.choice('n_neighbors', range(1, 100))
            }

        if self.adaptation == 'Peterfilter':
            adptdefault_value = {
                'eachCluster': 10
            }
            adptparamSpace = {
                'eachCluster': hp.choice('eachCluster', range(1, 100))
            }

        if self.adaptation == 'Universal':
            adptdefault_value = {
                'pvalue': 0.05,
                'QuantifyType': 'cliff'
            }
            adptparamSpace = {
                'pvalue': hp.uniform('pvalue', 0.01, 0.1),
                'QuantifyType': hp.choice('QuantifyType', ['cliff', 'cohen'])
            }

        if self.adaptation == 'DTB':
            adptdefault_value = {
                'DTBneighbors': 10,
                'DTBT': 20
            }
            adptparamSpace = {
                'DTBneighbors': hp.choice('DTBneighbors', range(1, 50)),
                'DTBT': hp.choice('DTBT', range(5, 30))
            }

        if self.adaptation == 'DS':
            adptdefault_value = {
                'DStopn': 5,
                'DSfss': 0.2
            }
            adptparamSpace = {
                'DStopn': hp.choice('DStopn', range(1, 15)),
                'DSfss': hp.uniform('DSfss', 0.1, 0.5)
            }

        if self.adaptation == 'DSBF':
            adptdefault_value = {
                'DSBFtopk': 1,
                'DSBFneighbors': 10
            }
            adptparamSpace = {
                'DSBFtopk': hp.choice('DSBFtopk', range(1, 10)),
                'DSBFneighbors': hp.choice('DSBFneighbors', range(1, 100))
            }


        if self.clf == 'Boost':
            clfdefault_value = {
                'Boostnestimator': 50,
                'BoostLearnrate': 1
            }
            clfparamSpace = {
                'Boostnestimator': hp.choice('Boostnestimator', range(10, 1000)),
                'BoostLearnrate': hp.uniform('BoostLearnrate', 0.01, 10)
            }

        if self.clf == 'RF':
            clfdefault_value = {
                'n_estimators': 10,
                'criterion': 'gini',
                'max_features': 'auto',
                'RFmin_samples_split': 2
            }
            clfparamSpace = {
                'n_estimators': hp.choice('n_estimators', range(10, 100)),
                'criterion': hp.choice('criterion', ['gini', 'entropy']),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                'RFmin_samples_split': hp.choice('RFmin_samples_split', range(2, int(len(self.sy) / 10)))
            }

        if self.clf == 'SVM':
            clfdefault_value = {
                'SVCkernel': {'kernel': 'poly', 'degree': 3, 'polycoef0': 0.0, 'polygamma': 1},
                'C': 1.0
            }
            clfparamSpace = {
                'SVCkernel': hp.choice('SVCkernel', [
                    {'kernel': 'linear'},
                    {'kernel': 'poly', 'degree': hp.choice('degree', range(1, 5)),
                     'polycoef0': hp.uniform('polycoef0', 0, 10),
                     'polygamma': hp.choice('polygamma', ["auto", "scale"])},
                    {'kernel': 'sigmoid', 'sigcoef0': hp.uniform('sigcoef0', 0, 10),
                     'siggamma': hp.choice('siggamma', ["auto", "scale"])},
                    {'kernel': 'rbf', 'rbfgamma': hp.choice('rbfgamma', ["auto", "scale"])}
                ]),
                'C': hp.uniform('C', 0.001, 1000),
            }

        if self.clf == 'NN':
            clfdefault_value = {
                'NNactive': 'relu',
                'NNalpha': 0.0001,
                'NNmaxiter': 200
            }
            clfparamSpace = {
                'NNactive': hp.choice('NNactive', ['identity', 'logistic', 'tanh', 'relu']),
                'NNalpha': hp.uniform('NNalpha', 1e-6, 1),
                'NNmaxiter': hp.choice('NNmaxiter', range(100, 1000))
            }

        if self.clf == 'KNN':
            clfdefault_value = {
                'KNNneighbors': 1
            }
            clfparamSpace = {
                'KNNneighbors': hp.choice('KNNneighbors', range(1, 50))
            }

        if self.clf == 'NB':
            clfdefault_value = {
                'NBType': 'gaussian'
            }
            clfparamSpace = {
                'NBType': hp.choice('NBType', ['gaussian', 'multinomial', 'bernoulli'])
            }

        if self.clf == 'Ridge':
            clfdefault_value = {
                'Ridgealpha': 1,
                'Ridgenormalize': False
            }
            clfparamSpace = {
                'Ridgealpha': hp.uniform('Ridgealpha', 0.001, 1000),
                'Ridgenormalize': hp.choice('Ridgenormlize', [True, False])
            }

        if self.clf == 'CART':
            clfdefault_value = {
                'criterion': 'gini',
                'max_features': 'auto',
                'CARTsplitter': 'best',
                'RFmin_samples_split': 2
            }
            clfparamSpace = {
                'criterion': hp.choice('criterion', ['gini', 'entropy']),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                'CARTsplitter': hp.choice('CARTsplitter', ['best', 'random']),
                'RFmin_samples_split': hp.choice('RFmin_samples_split', range(2, int(len(self.sy) / 10)))
            }

        paramSpace = dict(adptparamSpace, **clfparamSpace)
        default_value = dict(adptdefault_value, **clfdefault_value)
        self.def_value = self.objFunc(default_value)['result']
        best = fmin(self.objFunc, space=paramSpace, algo=tpe.suggest, max_evals=self.fe, trials=self.trails)

        his = dict()
        his['name'] = list(self.trails.trials[0]['misc']['vals'].keys())
        i = 0
        for item in self.trails.trials:
            results = list(deepflatten(item['misc']['vals'].values()))
            results.append(item['result']['result'])
            his[i] = results
            i += 1

        inc_value = self.trails.best_trial['result']['result']

        return np.asarray([self.def_value, inc_value]), his, best


class optParamAdpt(object):
    def __init__(self, sx, sy, tx, ty, loc, classifier, adaptation, fe=1000):
        self.sx = sx
        self.sy = sy
        self.tx = tx
        self.ty = ty
        self.loc = loc
        self.adaptation = adaptation
        self.clf = classifier
        self.fe = fe
        self.trails = Trials()

    def objFunc(self, params):
        self.p = cpdp(clf=self.clf, adpt=self.adaptation)
        self.p.set_params(**params)
        sx = self.sx
        sy = self.sy
        tx = self.tx
        ty = self.ty
        loc = self.loc
        res = self.p.run(sx, sy, tx, ty, loc)

        return {'loss': -np.mean(res), 'status': STATUS_OK, 'result': np.mean(res)}

    def run(self):
        trails = Trials()

        if self.adaptation == 'TCA':
            adptdefault_value = {
                'kernel_type': 'linear',
                'dim': 5,
                'lamb': 1,
                'gamma': 1
            }
            adptparamSpace = {
                'kernel_type': hp.choice('kernel_type', ['primal', 'linear', 'rbf', 'sam']),
                'dim': hp.choice('dim', range(5, max(self.sx.shape[1], self.tx.shape[1]))),
                'lamb': hp.uniform('lamb', 1e-6, 1e2),
                'gamma': hp.uniform('gamma', 1e-5, 1e2)
            }

        if self.adaptation == 'DBSCANfilter':
            adptdefault_value = {
                'eps': 1,
                'min_samples': 10
            }
            adptparamSpace = {
                'eps': hp.uniform('eps', 0.1, 1e2),
                'min_samples': hp.choice('min_samples', range(1, 100))
            }

        if self.adaptation == 'Bruakfilter':
            adptdefault_value = {
                'n_neighbors': 10
            }
            adptparamSpace = {
                'n_neighbors': hp.choice('n_neighbors', range(1, 100))
            }

        if self.adaptation == 'Peterfilter':
            adptdefault_value = {
                'eachCluster': 10
            }
            adptparamSpace = {
                'eachCluster': hp.choice('eachCluster', range(1, 100))
            }

        if self.adaptation == 'Universal':
            adptdefault_value = {
                'pvalue': 0.05,
                'QuantifyType': 'cliff'
            }
            adptparamSpace = {
                'pvalue': hp.uniform('pvalue', 0.01, 0.1),
                'QuantifyType': hp.choice('QuantifyType', ['cliff', 'cohen'])
            }

        if self.adaptation == 'HISNN':
            adptdefault_value = {
                'MinHam': 1.0,
                'HISNNneighbors': 5
            }
            adptparamSpace = {
                'MinHam': hp.uniform('MinHam', 0.5, 100),
                'HISNNneighbors': hp.choice('HISNNneighbors', range(1, 100))
            }

        if self.adaptation == 'DTB':
            adptdefault_value = {
                'DTBneighbors': 10,
                'DTBT': 20
            }
            adptparamSpace = {
                'DTBneighbors': hp.choice('DTBneighbors', range(1, 50)),
                'DTBT': hp.choice('DTBT', range(5, 30))
            }

        if self.adaptation == 'DS':
            adptdefault_value = {
                'DStopn': 5,
                'DSfss': 0.2
            }
            adptparamSpace = {
                'DStopn': hp.choice('DStopn', range(1, 15)),
                'DSfss': hp.uniform('DSfss', 0.1, 0.5)
            }

        if self.adaptation == 'DSBF':
            adptdefault_value = {
                'DSBFtopk': 1,
                'DSBFneighbors': 10
            }
            adptparamSpace = {
                'DSBFtopk': hp.choice('DSBFtopk', range(1, 10)),
                'DSBFneighbors': hp.choice('DSBFneighbors', range(1, 100))
            }

        self.def_value = self.objFunc(adptdefault_value)['result']
        best = fmin(self.objFunc, space=adptparamSpace, algo=tpe.suggest, max_evals=self.fe, trials=self.trails)
        his = dict()
        his['name'] = list(self.trails.trials[0]['misc']['vals'].keys())
        i = 0
        for item in self.trails.trials:
            results = list(deepflatten(item['misc']['vals'].values()))
            results.append(item['result']['result'])
            his[i] = results
            i += 1

        inc_value = self.trails.best_trial['result']['result']

        return np.asarray([self.def_value, inc_value]), his, best


class optParamCLF(object):
    def __init__(self, sx, sy, tx, ty, loc, classifier, adaptation,fe=1000):
        self.sx = sx
        self.sy = sy
        self.tx = tx
        self.ty = ty
        self.loc = loc
        self.adaptation = adaptation
        self.clf = classifier
        self.fe = fe
        self.trails = Trials()

    def objFunc(self, params):
        self.p = cpdp(clf=self.clf, adpt=self.adaptation)
        self.p.set_params(**params)
        sx = self.sx
        sy = self.sy
        tx = self.tx
        ty = self.ty
        loc = self.loc
        res = self.p.run(sx, sy, tx, ty, loc)

        return {'loss': -np.mean(res), 'status': STATUS_OK, 'result': np.mean(res)}

    def run(self):
        trails = Trials()

        if self.clf == 'Boost':
            clfdefault_value = {
                'Boostnestimator': 50,
                'BoostLearnrate': 1
            }
            clfparamSpace = {
                'Boostnestimator': hp.choice('Boostnestimator', range(10, 1000)),
                'BoostLearnrate': hp.uniform('BoostLearnrate', 0.01, 10)
            }

        if self.clf == 'RF':
            clfdefault_value = {
                'n_estimators': 10,
                'criterion': 'gini',
                'max_features': 'auto',
                'RFmin_samples_split': 2
            }
            clfparamSpace = {
                'n_estimators': hp.choice('n_estimators', range(10, 100)),
                'criterion': hp.choice('criterion', ['gini', 'entropy']),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                'RFmin_samples_split': hp.choice('RFmin_samples_split', range(2, int(len(self.sy) / 10)))
            }

        if self.clf == 'SVM':
            clfdefault_value = {
                'SVCkernel': {'kernel': 'poly', 'degree': 3, 'polycoef0': 0.0, 'polygamma': 1},
                'C': 1.0
            }
            clfparamSpace = {
                'SVCkernel': hp.choice('SVCkernel', [
                    {'kernel': 'linear'},
                    {'kernel': 'poly', 'degree': hp.choice('degree', range(1, 5)),
                     'polycoef0': hp.uniform('polycoef0', 0, 10),
                     'polygamma': hp.choice('polygamma', ["auto", "scale"])},
                    {'kernel': 'sigmoid', 'sigcoef0': hp.uniform('sigcoef0', 0, 10),
                     'siggamma': hp.choice('siggamma', ["auto", "scale"])},
                    {'kernel': 'rbf', 'rbfgamma': hp.choice('rbfgamma', ["auto", "scale"])}
                ]),
                'C': hp.uniform('C', 0.001, 1000),
            }

        if self.clf == 'NN':
            clfdefault_value = {
                'NNactive': 'relu',
                'NNalpha': 0.0001,
                'NNmaxiter': 200
            }
            clfparamSpace = {
                'NNactive': hp.choice('NNactive', ['identity', 'logistic', 'tanh', 'relu']),
                'NNalpha': hp.uniform('NNalpha', 1e-6, 1),
                'NNmaxiter': hp.choice('NNmaxiter', range(100, 1000))
            }

        if self.clf == 'KNN':
            clfdefault_value = {
                'KNNneighbors': 1
            }
            clfparamSpace = {
                'KNNneighbors': hp.choice('KNNneighbors', range(1, 50))
            }

        if self.clf == 'NB':
            clfdefault_value = {
                'NBType': 'gaussian'
            }
            clfparamSpace = {
                'NBType': hp.choice('NBType', ['gaussian', 'multinomial', 'bernoulli'])
            }

        if self.clf == 'Ridge':
            clfdefault_value = {
                'Ridgealpha': 1,
                'Ridgenormalize': False
            }
            clfparamSpace = {
                'Ridgealpha': hp.uniform('Ridgealpha', 0.001, 1000),
                'Ridgenormalize': hp.choice('Ridgenormlize', [True, False])
            }

        if self.clf == 'CART':
            clfdefault_value = {
                'criterion': 'gini',
                'max_features': 'auto',
                'CARTsplitter': 'best',
                'RFmin_samples_split': 2
            }
            clfparamSpace = {
                'criterion': hp.choice('criterion', ['gini', 'entropy']),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                'CARTsplitter': hp.choice('CARTsplitter', ['best', 'random']),
                'RFmin_samples_split': hp.choice('RFmin_samples_split', range(2, int(len(self.sy) / 10)))
            }

        self.def_value = self.objFunc(clfdefault_value)['result']
        best = fmin(self.objFunc, space=clfparamSpace, algo=tpe.suggest, max_evals=self.fe, trials=self.trails)

        his = dict()
        his['name'] = list(self.trails.trials[0]['misc']['vals'].keys())
        i = 0
        for item in self.trails.trials:
            results = list(deepflatten(item['misc']['vals'].values()))
            results.append(item['result']['result'])
            his[i] = results
            i += 1

        inc_value = self.trails.best_trial['result']['result']

        # print(def_value)
        return np.asarray([self.def_value, inc_value]), his, best


class optParamSEQ(object):
    def __init__(self, sx, sy, tx, ty, loc, classifier, adaptation,fe=1000):
        self.sx = sx
        self.sy = sy
        self.tx = tx
        self.ty = ty
        self.loc = loc
        self.adaptation = adaptation
        self.clf = classifier
        self.fe = fe
        self.trails = Trials()
        self.Atrails = Trials()

        self.SEQ = 0

    def objFunc(self, params):
        if self.SEQ == 1:
            params = dict(params, **self.Adptbest)
        self.p = cpdp(clf=self.clf, adpt=self.adaptation)
        self.p.set_params(**params)
        sx = self.sx
        sy = self.sy
        tx = self.tx
        ty = self.ty
        loc = self.loc
        res = self.p.run(sx, sy, tx, ty, loc)

        return {'loss': -np.mean(res), 'status': STATUS_OK, 'result': np.mean(res)}

    def run(self):
        Atrails = Trials()
        trails = Trials()

        if self.adaptation == 'TCA':
            adptdefault_value = {
                'kernel_type': 'linear',
                'dim': 5,
                'lamb': 1,
                'gamma': 1
            }
            adptparamSpace = {
                'kernel_type': hp.choice('kernel_type', ['primal', 'linear', 'rbf', 'sam']),
                'dim': hp.choice('dim', range(5, max(self.sx.shape[1], self.tx.shape[1]))),
                'lamb': hp.uniform('lamb', 1e-6, 1e2),
                'gamma': hp.uniform('gamma', 1e-5, 1e2)
            }

        if self.adaptation == 'DBSCANfilter':
            adptdefault_value = {
                'eps': 1,
                'min_samples': 10
            }
            adptparamSpace = {
                'eps': hp.uniform('eps', 0.1, 1e2),
                'min_samples': hp.choice('min_samples', range(1, 100))
            }

        if self.adaptation == 'Bruakfilter':
            adptdefault_value = {
                'n_neighbors': 10
            }
            adptparamSpace = {
                'n_neighbors': hp.choice('n_neighbors', range(1, 100))
            }

        if self.adaptation == 'Peterfilter':
            adptdefault_value = {
                'eachCluster': 10
            }
            adptparamSpace = {
                'eachCluster': hp.choice('eachCluster', range(1, 100))
            }

        if self.adaptation == 'Universal':
            adptdefault_value = {
                'pvalue': 0.05,
                'QuantifyType': 'cliff'
            }
            adptparamSpace = {
                'pvalue': hp.uniform('pvalue', 0.01, 0.1),
                'QuantifyType': hp.choice('QuantifyType', ['cliff', 'cohen'])
            }


        if self.adaptation == 'DTB':
            adptdefault_value = {
                'DTBneighbors': 10,
                'DTBT': 20
            }
            adptparamSpace = {
                'DTBneighbors': hp.choice('DTBneighbors', range(1, 50)),
                'DTBT': hp.choice('DTBT', range(5, 30))
            }

        if self.adaptation == 'DS':
            adptdefault_value = {
                'DStopn': 5,
                'DSfss': 0.2
            }
            adptparamSpace = {
                'DStopn': hp.choice('DStopn', range(1, 15)),
                'DSfss': hp.uniform('DSfss', 0.1, 0.5)
            }

        if self.adaptation == 'DSBF':
            adptdefault_value = {
                'DSBFtopk': 1,
                'DSBFneighbors': 10
            }
            adptparamSpace = {
                'DSBFtopk': hp.choice('DSBFtopk', range(1, 10)),
                'DSBFneighbors': hp.choice('DSBFneighbors', range(1, 100))
            }

        if self.clf == 'Boost':
            clfdefault_value = {
                'Boostnestimator': 50,
                'BoostLearnrate': 1
            }
            clfparamSpace = {
                'Boostnestimator': hp.choice('Boostnestimator', range(10, 1000)),
                'BoostLearnrate': hp.uniform('BoostLearnrate', 0.01, 10)
            }

        if self.clf == 'RF':
            clfdefault_value = {
                'n_estimators': 10,
                'criterion': 'gini',
                'max_features': 'auto',
                'RFmin_samples_split': 2
            }
            clfparamSpace = {
                'n_estimators': hp.choice('n_estimators', range(10, 100)),
                'criterion': hp.choice('criterion', ['gini', 'entropy']),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                'RFmin_samples_split': hp.choice('RFmin_samples_split', range(2, int(len(self.sy) / 10)))
            }

        if self.clf == 'SVM':
            clfdefault_value = {
                'SVCkernel': {'kernel': 'poly', 'degree': 3, 'polycoef0': 0.0, 'polygamma': 1},
                'C': 1.0
            }
            clfparamSpace = {
                'SVCkernel': hp.choice('SVCkernel', [
                    {'kernel': 'linear'},
                    {'kernel': 'poly', 'degree': hp.choice('degree', range(1, 5)),
                     'polycoef0': hp.uniform('polycoef0', 0, 10),
                     'polygamma': hp.choice('polygamma', ["auto", "scale"])},
                    {'kernel': 'sigmoid', 'sigcoef0': hp.uniform('sigcoef0', 0, 10),
                     'siggamma': hp.choice('siggamma', ["auto", "scale"])},
                    {'kernel': 'rbf', 'rbfgamma': hp.choice('rbfgamma', ["auto", "scale"])}
                ]),
                'C': hp.uniform('C', 0.001, 1000),
            }

        if self.clf == 'NN':
            clfdefault_value = {
                'NNactive': 'relu',
                'NNalpha': 0.0001,
                'NNmaxiter': 200
            }
            clfparamSpace = {
                'NNactive': hp.choice('NNactive', ['identity', 'logistic', 'tanh', 'relu']),
                'NNalpha': hp.uniform('NNalpha', 1e-6, 1),
                'NNmaxiter': hp.choice('NNmaxiter', range(100, 1000))
            }

        if self.clf == 'KNN':
            clfdefault_value = {
                'KNNneighbors': 1
            }
            clfparamSpace = {
                'KNNneighbors': hp.choice('KNNneighbors', range(1, 50))
            }

        if self.clf == 'NB':
            clfdefault_value = {
                'NBType': 'gaussian'
            }
            clfparamSpace = {
                'NBType': hp.choice('NBType', ['gaussian', 'multinomial', 'bernoulli'])
            }

        if self.clf == 'Ridge':
            clfdefault_value = {
                'Ridgealpha': 1,
                'Ridgenormalize': False
            }
            clfparamSpace = {
                'Ridgealpha': hp.uniform('Ridgealpha', 0.001, 1000),
                'Ridgenormalize': hp.choice('Ridgenormlize', [True, False])
            }

        if self.clf == 'CART':
            clfdefault_value = {
                'criterion': 'gini',
                'max_features': 'auto',
                'CARTsplitter': 'best',
                'RFmin_samples_split': 2
            }
            clfparamSpace = {
                'criterion': hp.choice('criterion', ['gini', 'entropy']),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                'CARTsplitter': hp.choice('CARTsplitter', ['best', 'random']),
                'RFmin_samples_split': hp.choice('RFmin_samples_split', range(2, int(len(self.sy) / 10)))
            }

        default_value = dict(adptdefault_value, **clfdefault_value)
        self.def_value = self.objFunc(default_value)['result']
        self.Adptbest = fmin(self.objFunc, space=adptparamSpace, algo=tpe.suggest, max_evals=int(self.fe * 0.5),
                             trials=self.Atrails)
        self.Adptbest = space_eval(adptparamSpace, self.Adptbest)

        his = dict()
        try:
            his['name'] = list(self.Atrails.trials[0]['misc']['vals'].keys()) + list(clfdefault_value.keys())
        except:
            his['name'] = [None]
        i = 0
        for item in self.Atrails.trials:
            if item['state'] == 2:
                results = list(deepflatten(item['misc']['vals'].values())) + list(clfdefault_value.values())
                results.append(item['result']['result'])
                his[i] = results
                i += 1

        self.SEQ = 1
        Clfbest = fmin(self.objFunc, space=clfparamSpace, algo=tpe.suggest, max_evals=int(self.fe * 0.5),
                       trials=self.trails)

        try:
            his['name1'] = list(self.Adptbest.keys()) + list(self.trails.trials[0]['misc']['vals'].keys())
        except:
            his['name1'] = [None]
        for item in self.trails.trials:
            if item['state'] == 2:
                results = list(self.Adptbest.values()) + list(deepflatten(item['misc']['vals'].values()))
                results.append(item['result']['result'])
                his[i] = results
                i += 1

        inc_value = self.trails.best_trial['result']['result']

        return np.asarray([self.def_value, inc_value]), his, Clfbest
