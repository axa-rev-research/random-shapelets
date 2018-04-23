import pandas, numpy
from multiprocessing import Pool
from itertools import repeat
import pyximport; pyximport.install()

from subsequence_transformation_cython import *

def get_distance(panel, candidate, panel_test=None):
    """
    Compute the distance between a shapelet candidate and a time series dataset.

    :param panel: Panel with a time series dataset: items are variables, major_axis is time and minor_axis are the time series
    :type panel: pandas.Panel
    :param candidate: Details of the shapelet candidate (see SearchSpace class)
    :type candidate: dictionary
    :param panel_test:
    :type panel_test: pandas.Panel
    :return: Rolling distances between time series and shapelet candidate
    :rtype: pandas.DataFrame
    """

    var = candidate['var']
    ts = candidate['ts']
    pos = candidate['pos']
    L = candidate['length']

    shpt = panel.ix[var,pos:pos+L,ts]

    if panel_test is None:
        df = panel.ix[var]
    else:
        df = panel_test.ix[var]

    res = df.apply(lambda x: pandas.Series(sqeuclidean(x.values, shpt.values)))
    return res


def get_aggregate(distances, candidate):
    """
    Compute the distance between a shapelet candidate and a time series dataset.

    :param distances: Dataframe with the distances between the shapelet candidate and the time series dataset
    :type distances: pandas.DataFrame
    :param candidate: Details of the shapelet candidate (see SearchSpace class)
    :type candidate: dictionary
    :return: Aggregation of the rolling distances between time series and shapelet candidate
    :rtype: pandas.DataFrame
    """

    magg = candidate['magg']
    agg = magg.split('+')[1]
    name = candidate['name']
    
    if agg == 'min':
        res = distances.min()
    elif agg == 'max':
        res = distances.max()

    # TODO: add other distribution stats, number of peaks, etc.

    res1 = distances.min()
    res1.name = name+'#Min'
    res2 = distances.idxmin()
    res2.name = name+'#Argument'
    res = pandas.concat((res2,res1), axis=1)

    return res

def get_east_transform_sub(args):
        panel_train = args[0]
        candidate = args[1]
        panel_test = args[2]
        distances = get_distance(panel_train, candidate, panel_test=panel_test)
        return get_aggregate(distances, candidate)

def get_east_transform(panel_train, candidates, panel_test=None, n_jobs=1):
    args = zip(repeat(panel_train), candidates, repeat(panel_test))
    with Pool(n_jobs) as pool:
        res = pool.map(get_east_transform_sub, args)
    
    return pandas.concat((r for r in res), axis=1)

class SearchSpace:

    def __init__(self, panel, metric_agg=['sqeuclidean+min'], minL=None, maxL=None):
        """
        Instantiate the Search Space
        :param panel: Training set with variables as items, time series as minor_axis and time as major_axis
        :type panel: pandas.Panel
        :param metric_agg: List of distance metrics + aggregation functions to perform the subsequence transformation
        :type metric_agg: list
        """

        self.panel = panel
        self.df_n_candidates = pandas.DataFrame(index=panel.minor_axis, columns=panel.items)
        for var in self.panel.items:
            for ts in self.panel.minor_axis:
                L = self.panel.ix[var,:,ts].shape[0]
                self.df_n_candidates.ix[ts,var] = 0.5*L*(L+1)

        # Store metric+aggregation
        self.metric_agg = metric_agg

        # Compute the number of shapelet candidates by variable & in total
        self.n_candidates_by_var = self.df_n_candidates.sum(axis=0)
        self.n_candidates_total = self.n_candidates_by_var.sum()

        # Compute the bias to apply to each variable for the drawing
        self.proba_by_var = self.n_candidates_by_var / self.n_candidates_total
        
        self.minL = minL
        self.maxL = maxL

    def draw_candidate(self):
        """
        Draw a shapelet candidate in an instantiated SearchSpace

        :return: Shapelet candidate details
        :rtype: list of dictionary
        """

        # Draw variable
        var = numpy.random.choice(self.panel.items, p=self.proba_by_var)

        # Draw time series
        proba_by_ts = self.df_n_candidates.ix[:,var] / self.df_n_candidates.ix[:,var].sum()
        ts = numpy.random.choice(self.panel.minor_axis, p=numpy.double(proba_by_ts.values))

        # Draw candidate length
        ts_length = self.panel.ix[var,:,ts].shape[0]
        n_candidates_by_L = [(ts_length-L+1) for L in range(1, ts_length)]
        proba_by_L = n_candidates_by_L / numpy.sum(n_candidates_by_L)
        L = numpy.random.choice(range(1, ts_length), p=proba_by_L)

        # Draw starting position
        pos = numpy.random.choice(range(self.panel.ix[var,:,ts].shape[0]-L))

        # Draw metric+agg
        magg = numpy.random.choice(self.metric_agg)

        return [{'var':var, 'ts':ts, 'length':L, 'pos':pos, 'magg':magg, 'name':str(var)+'#'+str(ts)+'#'+str(pos)+'-'+str(L)+'#'+str(magg)}]


    def draw_candidates(self, n=10):
        """
        Draw several shapelet candidates in an instantiated SearchSpace

        :param n: Number of shapelet candidates to draw
        :type n: int
        :return: Details of the shapelet candidates
        :rtype: list of dictionary
        """

        candidates = []
        while len(candidates)<n:
            candidate_tmp = self.draw_candidate()
            # Check whether candidate has not already been drawn
            if candidate_tmp not in candidates:
                if (self.minL is None and self.maxL is None) or (self.minL<candidate_tmp[0]['length']<self.maxL):
                    candidates += candidate_tmp

        return candidates


def get_candidate(panel, candidate):
    """
    Return the time series correspondant to the parameter "candidate"
    """
    var = candidate['var']
    ts = candidate['ts']
    pos = candidate['pos']
    L = candidate['length']
    return panel.ix[var,pos:pos+L,ts]

def generate_dummy_timeseries():
    """
    Generate a random time series dataset
    """
    tmp = {str(i):pandas.DataFrame([numpy.random.random(1000) for _ in range(1000)]).T for i in range(10)}
    return pandas.Panel(tmp)
