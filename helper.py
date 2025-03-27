import pandas as pd
import statsmodels.api as sm

def ordinary_lest_squares_regression(y,X):
    model = sm.OLS(endog=y,exog=sm.add_constant(X))
    results = model.fit()

    print(results.summary())

    df_residual = pd.DataFrame(results.resid,columns=["error"])
    plot=sm.qqplot(df_residual,line="s")
    plot.show()

    return results,df_residual