#
import pandas

#


#
def portfolio_decision_dynamics_simple(k0, opened, closed, decision):

    opened = pandas.DataFrame(opened)
    closed = pandas.DataFrame(closed)
    decision = pandas.DataFrame(decision)

    opened['dindunuffin'] = 1.0
    closed['dindunuffin'] = 1.0

    def checky(x):
        if x.sum() == 0:
            return 1.0
        else:
            return 0.0

    decision['dindunuffin'] = decision.apply(func=checky, axis=1)

    effect = closed / opened

    k_array = [k0]
    for j in range(decision.shape[0]):
        k_new = (decision.values[j, :] * effect.values[j, :] * k_array[-1]).sum()
        k_array.append(k_new)
    result = pandas.Series(data=k_array[1:], index=decision.index, name='PortfolioDynamics')

    return result
