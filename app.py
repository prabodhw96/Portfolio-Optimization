import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import pandas_datareader.data as web
from datetime import date, timedelta
import plotly.graph_objects as go
from scipy import stats
import scipy.optimize as sco
import streamlit as st

STOCK_URL = (
	"nasdaq_stocks.csv"
)

@st.cache(allow_output_mutation=True)
def load_data():
	df = pd.read_csv(STOCK_URL)
	return df

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def fetch_data(stocks):
	try:
		data = web.DataReader(stocks, data_source="yahoo",start=date.today()-timedelta(days=365))["Adj Close"]
	except:
		st.success("Please select stocks :)")
		st.stop()
	return data

def coskew(df):
	num = len(df.columns)
	mtx = np.zeros(shape = (len(df), num**2))
	v = df.values
	means = v.mean(0,keepdims=True)
	v1 = (v-means).T

	for i in range(num):
		for j in range(num):
			vals = v1[i]*v1[j]
			mtx[:,(i*num)+j] = vals/float((len(df)-1)*df.iloc[:,i].std()*df.iloc[:,j].std())

	m3 = np.dot(v1,mtx)

	for i in range(num**2):
		use = i%num
		m3[:,i] = m3[:,i]/float(df.iloc[:,use].std())

	return m3

def cokurt(df):
	num = len(df.columns)
	mtx1 = np.zeros(shape = (len(df), num**2))
	mtx2 = np.zeros(shape = (len(df), num**3))
	v = df.values
	means = v.mean(0,keepdims=True)
	v1 = (v-means).T

	for k in range(num):
		for i in range(num):
			for j in range(num):
				vals = v1[i]*v1[j]*v1[k]
				mtx2[:,(k*(num**2))+(i*num)+j] = vals/float((len(df)-1)*df.iloc[:,i].std()*df.iloc[:,j].std()*df.iloc[:,k].std())

	m4 = np.dot(v1,mtx2)

	for i in range(num**3):
		use = i%num
		m4[:,i] = m4[:,i]/float(df.iloc[:,use].std())

	return m4

def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
	returns = np.sum(mean_returns*weights)*252
	std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))*np.sqrt(252)
	return std, returns

def neg_prob_sharpe_ratio(weights, mean_returns, log_returns, cov_matrix, risk_free_rate, sharpe_ratio_benchmark):
	p_var, p_ret = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
	sharpe_ratio = (p_ret - risk_free_rate)/p_var
	sharpe_ratio_stdev = np.sqrt((1+(0.5*sharpe_ratio**2) - (coskew(log_returns).mean()*sharpe_ratio) + (((cokurt(log_returns).mean()-3)/4)*sharpe_ratio**2))/(len(mean_returns)-1))
	prob_sharpe_ratio = stats.norm.cdf((sharpe_ratio - sharpe_ratio_benchmark)/sharpe_ratio_stdev)
	return -1*prob_sharpe_ratio

def max_sharpe_ratio(mean_returns, log_returns, cov_matrix, risk_free_rate, sharpe_ratio_benchmark=0.0):
	num_assets = len(mean_returns)
	args = (mean_returns, log_returns, cov_matrix, risk_free_rate, sharpe_ratio_benchmark)
	constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
	bound = (0.0,1.0)
	bounds = tuple(bound for asset in range(num_assets))
	result = sco.minimize(neg_prob_sharpe_ratio, num_assets*[1./num_assets,], args=args, method="SLSQP", bounds=bounds, constraints=constraints)
	return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
	return portfolio_annualized_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
	num_assets = len(mean_returns)
	args = (mean_returns, cov_matrix)
	constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
	bound = (0.0, 1.0)
	bounds = tuple(bound for asset in range(num_assets))
	result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method="SLSQP", bounds=bounds, constraints=constraints)
	return result

def efficient_return(mean_returns, cov_matrix, target):
	num_assets = len(mean_returns)
	args = (mean_returns, cov_matrix)

	def portfolio_return(weights):
		return portfolio_annualized_performance(weights, mean_returns, cov_matrix)[1]

	constraints = ({"type": "eq", "fun": lambda x: portfolio_return(x) - target}, {"type": "eq", "fun": lambda x: np.sum(x) - 1})
	bounds = tuple((0,1) for asset in range(num_assets))
	result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method="SLSQP", bounds=bounds, constraints=constraints)
	return result

def efficient_frontier(mean_returns, cov_matrix, returns_range):
	efficients = []
	for ret in returns_range:
		efficients.append(efficient_return(mean_returns, cov_matrix, ret))
	return efficients

def cokurt(df):
	num = len(df.columns)
	mtx1 = np.zeros(shape = (len(df), num**2))
	mtx2 = np.zeros(shape = (len(df), num**3))
	v = df.values
	means = v.mean(0,keepdims=True)
	v1 = (v-means).T

	for k in range(num):
		for i in range(num):
			for j in range(num):
				vals = v1[i]*v1[j]*v1[k]
				mtx2[:,(k*(num**2))+(i*num)+j] = vals/float((len(df)-1)*df.iloc[:,i].std()*df.iloc[:,j].std()*df.iloc[:,k].std())

	m4 = np.dot(v1,mtx2)
	for i in range(num**3):
		use = i%num
		m4[:,i] = m4[:,i]/float(df.iloc[:,use].std())

	return m4


st.title("Optimal Asset Allocation")

symbol = load_data()
symbol_dict = dict(zip(symbol["Name"], symbol["Symbol"]))

st.sidebar.markdown("# Search by")
search = st.sidebar.radio("", ("Name", "Symbol"))

st.markdown("### Select stocks")
if search == "Symbol":
	stocks = st.multiselect("", list(symbol["Symbol"]), ["AAPL", "AMZN", "GOOG", "NFLX", "TSLA"])
else:
	stocks = st.multiselect("", list(symbol["Name"]), ["Apple Inc.", "Amazon.com Inc.", "Alphabet Inc.", "Netflix Inc.", "Tesla Inc."])
	stocks = [symbol_dict[i] for i in stocks]

data = fetch_data(stocks)

fig_lg = go.Figure()
idx = 0
for i in data.columns:
	fig_lg.add_trace(go.Scatter(x=data.index, y=data[stocks[idx]], name=stocks[idx]))
	idx += 1
fig_lg.update_layout(yaxis_title="Price in $")

returns = data.pct_change()
fig_dr = go.Figure()
idx = 0
for i in returns.columns:
	fig_dr.add_trace(go.Scatter(x=returns.index, y=returns[stocks[idx]], name=stocks[idx]))
	idx += 1
fig_dr.update_layout(yaxis_title="Daily returns")

st.sidebar.markdown("# Show Plots")

if st.sidebar.checkbox("Line Graph", False):
	st.plotly_chart(fig_lg)

if st.sidebar.checkbox("Daily Returns", False):
	st.plotly_chart(fig_dr)

mean_returns = returns.mean()
cov_matrix = returns.cov()
risk_free_rate = 0.0178
log_returns = np.log(data/data.shift(1))
log_returns = log_returns.dropna()

log_ret = np.log(data/data.shift(1))
log_ret = log_ret.dropna()

max_sharpe = max_sharpe_ratio(mean_returns, log_returns, cov_matrix, risk_free_rate, sharpe_ratio_benchmark=2.0)
sdp, rp = portfolio_annualized_performance(max_sharpe["x"], mean_returns, cov_matrix)
max_sharpe_allocation = pd.DataFrame(max_sharpe["x"], index=data.columns, columns=["allocation"])
max_sharpe_allocation["allocation"] = [round(i*100, 2)for i in max_sharpe_allocation["allocation"]]
max_sharpe_allocation = max_sharpe_allocation.T

min_vol = min_variance(mean_returns, cov_matrix)
sdp_min, rp_min = portfolio_annualized_performance(min_vol["x"], mean_returns, cov_matrix)
min_vol_allocation = pd.DataFrame(min_vol.x,index=data.columns,columns=["allocation"])
min_vol_allocation["allocation"] = [round(i*100,2)for i in min_vol_allocation["allocation"]]
min_vol_allocation = min_vol_allocation.T

an_vol = np.std(returns)*np.sqrt(252)
an_rt = mean_returns*252

st.write("### Maximum Sharpe Ratio Portfolio Allocation")
st.write("Annualized Return:", round(rp, 2))
st.write("Annualized Volatility:", round(sdp, 2))
st.table(max_sharpe_allocation)

st.write("### Minimum Volatility Portfolio Allocation")
st.write("Annualized Return:", round(rp_min, 2))
st.write("Annualized Volatility:", round(sdp_min, 2))
st.table(min_vol_allocation)

st.write("### Individual Stock Returns and Volatility")
for i, txt in enumerate(data.columns):
	st.write("**{}:** Annualized Return: {}; Annualized Volatility: {}".format(txt, round(an_rt[i], 2), round(an_vol[i], 2)))

if st.sidebar.checkbox("Compare Portfolios", False):
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.scatter(an_vol, an_rt, marker="o", color="#f8766d", s=200)

	for i, txt in enumerate(data.columns):
		ax.annotate(txt, (an_vol[i], an_rt[i]), xytext=(10, 0), textcoords="offset points")

	ax.scatter(sdp, rp, marker="*", color="#01b0f6", s=500, label="Maximum Sharpe ratio")
	ax.scatter(sdp_min, rp_min, marker="*", color="#02bf7d", s=500, label="Minimum volatility")

	#target = np.linspace(rp_min, 0.34, 50)
	#efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)

	#ax.plot([pf["fun"] for pf in efficient_portfolios], target, linestyle="-.", color="black", label="efficient frontier")
	ax.set_title("Portfolio Optimization with Individual Stocks")
	ax.set_xlabel("annualized volatility")
	ax.set_ylabel("annualized returns")
	ax.legend(labelspacing=0.8)
	st.pyplot(fig)