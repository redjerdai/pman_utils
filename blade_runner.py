#
import numpy
import pandas


#
from pman_utils.portfolio_skeleton import PortfolioSleketon
from pman_utils.utils import portfolio_decision_dynamics_simple


#
M = 100
time_axis = pandas.date_range(start='2020-01-01', end='2021-01-01', freq='D')
N = time_axis.shape[0]

tickers = ['A{0}'.format(j) for j in numpy.arange(M)]
real = pandas.DataFrame(data=numpy.concatenate([numpy.arange(start=1000, step=10, stop=1000 + 10 * N).reshape(-1, 1) for _ in numpy.arange(M)], axis=1), index=time_axis, columns=tickers)
opens = real - 1

predicted_data = real.values + numpy.random.normal(loc=0.0, scale=1.0, size=(N, M))
predicted = pandas.DataFrame(data=predicted_data, index=time_axis, columns=tickers)

ptf = PortfolioSleketon(real, predicted)

ptf.compute()

dec, tp, sl = ptf.get_all()

res = portfolio_decision_dynamics_simple(k0=1000, opened=opens, closed=real, decision=dec)
