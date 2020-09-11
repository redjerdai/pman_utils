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
    def compute(self, top_q=0.9, min_yield=0.01):
        pct = self.predicted / self.real

        def quantile(x):
            mask = x >= min_yield
            if mask.sum() > 0:
                result = numpy.quantile(a=x.values, q=top_q)
            else:
                result = numpy.nan
            return result

        pct['_tHRESH'] = pct.apply(func=quantile, axis=1)

        def count(x):
            summed = (x[[x for x in x.index if x not in ['_tHRESH']]].values > x['_tHRESH']).sum()
            if summed > 0:
                result = 1 / summed
            else:
                result = 0
            return result

        pct['_pART'] = pct.apply(func=count, axis=1)

        cols = [x for x in pct.columns if x not in ['_tHRESH', '_pART']]

        self.decisions = pandas.DataFrame(data=numpy.zeros(shape=pct[cols].shape), columns=cols, index=pct.index)

        for column in cols:
            self.decisions.loc[pct[column] > pct['_tHRESH'], column] = pct.loc[pct[column] > pct['_tHRESH'], '_pART']

    def get_decisions(self):
        return self.decisions

    def get_stop_losses(self, loss_rate=0.1):
        losses_taken = self.real * (1 - loss_rate)
        return losses_taken

    def get_take_profits(self):
        data_ = numpy.full(shape=self.predicted.shape, fill_value=numpy.nan)
        profits_taken = pandas.DataFrame(data=data_, columns=self.predicted.columns.values, index=self.predicted.index)
        for col in profits_taken.columns.values:
            mask = self.decisions[col] != 0
            profits_taken.loc[mask, col] = self.predicted.loc[mask, col]
        return profits_taken

    def get_all(self):
        return self.get_decisions(), self.get_take_profits(), self.get_stop_losses()
