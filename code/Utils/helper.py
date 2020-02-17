from iteration_utilities import deepflatten
from Utils.Hyperopt_doer import optParamCLF, optParamSEQ, optParamAll, optParamAdpt
from Utils.File import *
import re, time
import numpy as np
from func_timeout import FunctionTimedOut
from sklearn.model_selection import train_test_split


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


def GetData(filename, showType=False):
    if 'JURECZKO' in filename:
        with open(filename, 'r') as f:
            data = f.readlines()
        x = []
        y = []
        empty = []

        # get the types of metrics from first line
        type = data[0].strip().split(';')
        type.pop()
        type.pop(0)

        # get the detail data of metrics
        for line in data[1:]:
            tmp = []

            odom = line.strip().split(';')
            # delete the project information
            for i in range(3):
                odom.pop(0)

            for i in range(len(odom)):
                if is_number(odom[i]):
                    tmp.append(float(odom[i]))
                else:
                    if i not in empty:
                        empty.append(i)
                    tmp.append(0)

            if tmp.pop() > 0:
                y.append(1)
            else:
                y.append(-1)
            x.append(tmp)

        x = np.delete(np.asarray(x), empty, axis=1)
        empty = sorted(empty)
        for i in range(len(empty)):
            type.pop(empty[len(empty) - i - 1])

        if showType:
            return x, np.asarray(y), type
        else:
            return x, np.asarray(y)

    else:
        with open(filename, 'r') as f:
            data = f.readlines()  # txt中所有字符串读入data
            x = []
            y = []
            type = []

            for line in data:
                if '###' in line:
                    odom = line.strip().split(' ')
                    odom.remove('###')
                    type = odom
                else:
                    tmp = []
                    odom = line.strip().split(',')  # 将单个数据分隔开存好
                    if not is_number(odom[0]):
                        continue
                    for item in odom:
                        if is_number(item):
                            tmp.append(float(item))
                        elif (item == 'true') or (item == 'TRUE') or (item == 'Y') or (item == 'buggy'):
                            y.append(1)
                        else:
                            y.append(0)
                    x.append(tmp)

        if showType:
            return np.asanyarray(x), np.asarray(y), type
        else:
            return np.asanyarray(x), np.asarray(y)


def SfindCommonMetric(fsource, ftarget, showDiff=False, showType=False):
    sx, sy, Stype = GetData(fsource, showType=True)
    tx, ty, Ttype = GetData(ftarget, showType=True)

    common = []

    ss = sx.shape
    tt = tx.shape
    for i in range(ss[1]):
        if Stype[i] in Ttype:
            common.append(Stype[i])

    if len(common) > 0:
        fsx = np.zeros((ss[0], len(common)))
        ftx = np.zeros((tt[0], len(common)))
        for i in range(len(common)):
            index = Stype.index(common[i])
            fsx[:, i] = sx[:, index]

            index = Ttype.index(common[i])
            ftx[:, i] = tx[:, index]

        DiffSx = np.zeros((ss[0], ss[1] - len(common)))
        DiffTx = np.zeros((tt[0], tt[1] - len(common)))

        i = 0
        for j in range(ss[1]):
            if Stype[j] not in common:
                DiffSx[:, i] = sx[:, j]
                i = i + 1
        i = 0
        for j in range(tt[1]):
            if Ttype[j] not in common:
                DiffTx[:, i] = tx[:, j]
                i = i + 1
        if showDiff and showType:
            return fsx, sy, ftx, ty, DiffSx, DiffTx, common
        elif showDiff and (not showType):
            return fsx, sy, ftx, ty, DiffSx, DiffTx
        elif (not showDiff) and showType:
            return fsx, sy, ftx, ty, common
        else:
            return fsx, sy, ftx, ty
    else:
        return 0, 0, 0, 0


def MfindCommonMetric(list, ftarget, split=False):
    tx, ty, Ttype = GetData(ftarget, showType=True)
    tt = tx.shape
    common = []

    flist = list.copy()
    ### find the common metric
    first = 1
    dump = []

    for item in flist:
        x, y, Stype = GetData(item, showType=True)
        ss = x.shape

        if first == 1:
            for i in range(ss[1]):
                if Stype[i] in Ttype:
                    common.append(Stype[i])
            first = 0
        else:
            for i in range(len(common)):
                if common[i] not in Stype and i not in dump:
                    dump.append(i)
    dump = sorted(dump, reverse=True)
    for i in range(len(dump)):
        common.pop(dump[i])

    ### read the data and concatendate

    if len(common) == 0:
        return 0, 0, 0, 0, []
    else:
        ftx = np.zeros((tt[0], len(common)))
        for i in range(len(common)):
            index = Ttype.index(common[i])
            ftx[:, i] = tx[:, index]

        sx, sy, Stype = GetData(flist.pop(), showType=True)

        fsx = np.zeros((len(sy), len(common)))
        for i in range(len(common)):
            index = Stype.index(common[i])
            fsx[:, i] = sx[:, index]

        loc = []
        base = 0

        for item in list:
            x, y, Type = GetData(item, showType=True)
            loc.append(base)
            base += len(y)
            fx = np.zeros((len(y), len(common)))
            for i in range(len(common)):
                index = Type.index(common[i])
                fx[:, i] = x[:, index]
            fsx = np.concatenate((fsx, fx), axis=0)
            sy = np.concatenate((sy, y), axis=0)

        if split:
            return fsx, sy, ftx, ty, loc
        else:
            return fsx, sy, ftx, ty, []


def GetDataList(flist):
    a = flist.pop()
    xs, ys, xt, yt, loc = MfindCommonMetric(flist, a)
    x = np.concatenate((xs, xt), axis=0)
    y = np.concatenate((ys, yt), axis=0)
    return x, y


def collectData(fname):
    count = len(open(fname, 'r').readlines())
    with open(fname, 'r') as f:
        tmp = list(map(eval, f.readline()[1:-2].split()))
        res = np.zeros((count - 1, len(tmp)))
        i = 0
        print(fname, len(tmp))
        for line in f:
            line = line[1:-2]
            res[i] = np.asarray(line.split())[:len(tmp)]
            i += 1
            print(np.asarray(line.split()))
    return np.concatenate(([tmp], res))


def resCollect():
    for mode in {'all', 'clf', 'seq', 'adpt'}:
        resDir = 'res' + mode.upper()
        for root, dirs, files in os.walk(resDir):
            for name in files:
                fname = os.path.join(root, name)
                result = collectData(fname)
                np.savetxt(fname, result, fmt='%.7f')


def normal(xs, xt):
    ss = xs.shape
    tt = xt.shape

    # normalization for source data
    res = np.zeros((ss[0], ss[1]))
    for i in range(ss[1]):
        tmp = xs[:, i]
        minm = np.min(tmp)
        maxm = np.max(tmp)
        res[:, i] = (tmp - minm) / (maxm - minm)
    xs = res

    # normalization for target data
    res = np.zeros((tt[0], tt[1]))
    for i in range(tt[1]):
        tmp = xt[:, i]
        minm = np.min(tmp)
        maxm = np.max(tmp)
        res[:, i] = (tmp - minm) / (maxm - minm)
    xt = res

    return xs, xt


# REPEAT denotes the number of repetition of parameter optimization
def RunExperiment(Xsource, Lsource, Xtarget, Ltarget, loc, target, adaptation, classifier, mode='adpt', repeat=10,
                  fe=1000):
    fres = create_dir('res' + mode.upper() + '/' + target)
    fp = create_dir('para' + mode.upper() + '/' + target)

    if adaptation in ['NNfilter', 'MCWs']:
        # supervised (repeat 10 times)
        for it in range(10):
            train, test, ytrain, ytest = train_test_split(Xtarget, Ltarget, test_size=0.9, random_state=42)

            num = repeat
            res = np.zeros((num, 2))
            for i in range(num):
                if mode == 'adpt':
                    optimizer = optParamAdpt(Xsource, Lsource, test, ytest, loc, classifier, adaptation, fe,
                                             train=train, Ltrain=ytrain)
                elif mode == 'all':
                    optimizer = optParamAll(Xsource, Lsource, test, ytest, loc, classifier, adaptation, fe, train=train,
                                            Ltrain=ytrain)
                elif mode == 'seq':
                    optimizer = optParamSEQ(Xsource, Lsource, test, ytest, loc, classifier, adaptation, fe, train=train,
                                            Ltrain=ytrain)
                elif mode == 'clf':
                    optimizer = optParamCLF(Xsource, Lsource, test, ytest, loc, classifier, adaptation, fe, train=train,
                                            Ltrain=ytrain)

                try:
                    res[i], his, best = optimizer.run()
                except FunctionTimedOut:
                    trails = optimizer.trails
                    if mode == 'seq':
                        if len(optimizer.trails.results) == 0:
                            trails = optimizer.Atrails
                    his = dict()
                    try:
                        his['name'] = list(trails.trials[0]['misc']['vals'].keys())
                    except:
                        his['name'] = [None]
                    j = 0
                    for item in trails.trials:
                        # only record the results of successful runs
                        if item['state'] == 2:
                            results = list(deepflatten(item['misc']['vals'].values()))
                            results.append(item['result']['result'])
                            his[j] = results
                            j += 1
                    if j > 0:
                        inc_value = trails.best_trial['result']['result']
                        res[i] = np.asarray([optimizer.def_value, inc_value])
                        best = trails.best_trial['misc']['vals']
                    else:
                        try:
                            inc_value = trails.best_trial['result']['result']
                            res[i] = np.asarray([optimizer.def_value, inc_value])
                            best = trails.best_trial['misc']['vals']
                        except:
                            res[i] = np.asarray([0, 0])
                            best = []

                # print the result into file
                with open(fres + adaptation + '-' + classifier + '.txt', 'a+') as f:
                    print(res[i], best, file=f)
                    if i == num - 1:
                        print('-----------------------------------------', file=f)

                with open(fp + adaptation + '-' + classifier + '-' + '.txt', 'a+') as f:
                    for item in his.values():
                        print(item, file=f)
                        if i == num - 1:
                            print('-----------------------------------------', file=f)

    else:
        # un-supervised (just 1 times)
        num = repeat
        res = np.zeros((num, 2))
        for i in range(num):
            if mode == 'adpt':
                optimizer = optParamAdpt(Xsource, Lsource, Xtarget, Ltarget, loc, classifier, adaptation, fe)
            elif mode == 'all':
                optimizer = optParamAll(Xsource, Lsource, Xtarget, Ltarget, loc, classifier, adaptation, fe)
            elif mode == 'seq':
                optimizer = optParamSEQ(Xsource, Lsource, Xtarget, Ltarget, loc, classifier, adaptation, fe)
            elif mode == 'clf':
                optimizer = optParamCLF(Xsource, Lsource, Xtarget, Ltarget, loc, classifier, adaptation, fe)

            try:
                res[i], his, best = optimizer.run()
            except FunctionTimedOut:
                trails = optimizer.trails
                if mode == 'seq':
                    if len(optimizer.trails.results) == 0:
                        trails = optimizer.Atrails
                his = dict()
                try:
                    his['name'] = list(trails.trials[0]['misc']['vals'].keys())
                except:
                    his['name'] = [None]
                j = 0
                for item in trails.trials:
                    # only record the results of successful runs
                    if item['state'] == 2:
                        results = list(deepflatten(item['misc']['vals'].values()))
                        results.append(item['result']['result'])
                        his[j] = results
                        j += 1
                if j > 0:
                    inc_value = trails.best_trial['result']['result']
                    res[i] = np.asarray([optimizer.def_value, inc_value])
                    best = trails.best_trial['misc']['vals']
                else:
                    try:
                        inc_value = trails.best_trial['result']['result']
                        res[i] = np.asarray([optimizer.def_value, inc_value])
                        best = trails.best_trial['misc']['vals']
                    except:
                        res[i] = np.asarray([0, 0])
                        best = []


            # print the result into file
            with open(fres + adaptation + '-' + classifier + '.txt', 'a+') as f:
                print(res[i], best, file=f)

            with open(fp + adaptation + '-' + classifier + '.txt', 'a+') as f:
                for item in his.values():
                    print(item, file=f)
