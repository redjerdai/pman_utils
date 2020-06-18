#
import numpy
import pandas


class PortfolioSleketon:

    def __init__(self, real, predicted):
        """
        Both for real and predited:
        They are to be DataFrames
        Each column (1d column axis, not multiindex) to correspond to an instrument
        Index is to be 1d (not multiindex) and to correspond to time axis
        For the same time step values at real represent the last available prices, and at predicted the last available prediction
        for the next time step
        """

        self.real = real.copy()
        self.predicted = predicted.copy()
        self.decisions = None
        self.composite = None

    # now we just invest in top 10% party
    def compute(self):
        pct = self.predicted / self.real

        def quantile(x):
            return numpy.quantile(a=x.values, q=0.9)

        pct['_tHRESH'] = pct.apply(func=quantile, axis=1)

        def count(x):
            return 1 / (x[[x for x in x.index if x not in ['_tHRESH']]].values > x['_tHRESH']).sum()

        pct['_pART'] = pct.apply(func=count, axis=1)

        cols = [x for x in pct.columns if x not in ['_tHRESH', '_pART']]

        self.decisions = pandas.DataFrame(data=numpy.zeros(shape=pct[cols].shape), columns=cols, index=pct.index)

        for column in cols:
            self.decisions.loc[pct[column] > pct['_tHRESH'], column] = pct.loc[pct[column] > pct['_tHRESH'], '_pART']

    def get_decisions(self):
        return self.decisions
