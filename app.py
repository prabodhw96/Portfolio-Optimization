import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
from datetime import date, timedelta
import plotly.graph_objects as go
from scipy import stats
import scipy.optimize as sco
import dcor
import networkx as nx
import random
import streamlit as st

STOCK_URL = (
	"nasdaq_stocks.csv"
)

@st.cache(allow_output_mutation=True)
def load_data():
	df = pd.read_csv(STOCK_URL)
	return df

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def fetch_data(stocks, days=365, backtest=False):
	try:
		if backtest == False:
			data = pdr.get_data_yahoo(stocks, start=date.today()-timedelta(days=days))["Adj Close"]
		else:
			data = pdr.get_data_yahoo(stocks, start=date.today()-timedelta(days=days), end=date.today()-timedelta(365))["Adj Close"]
	except:
		st.success("Please select stocks :)")
		st.stop()
	return data

def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
	returns = np.sum(mean_returns*weights)*252
	std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))*np.sqrt(252)
	return std, returns

def neg_prob_sharpe_ratio(weights, mean_returns, log_returns, cov_matrix, risk_free_rate, sharpe_ratio_benchmark):
	p_var, p_ret = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
	sharpe_ratio = (p_ret - risk_free_rate)/p_var
	coskew = pd.Series(stats.skew(mean_returns), index=mean_returns.index).mean()
	cokurt = pd.Series(stats.kurtosis(mean_returns, fisher=False), index=mean_returns.index).mean()    
	sharpe_ratio_stdev = np.sqrt((1+(0.5*sharpe_ratio**2) - coskew*sharpe_ratio) + (((cokurt-3)/4)*sharpe_ratio**2))/(len(mean_returns)-1)
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

def correlation_network(df, stocks):
	df = df.diff().dropna()
	df_dcor = pd.DataFrame(index=stocks, columns=stocks)
	k = 0
	for i in stocks:
		v_i = df.loc[:, i].values
		for j in stocks[k:]:
			v_j = df.loc[:, j].values
			dcor_val = dcor.distance_correlation(v_i, v_j)
			df_dcor.at[i,j] = dcor_val
			df_dcor.at[j,i] = dcor_val
		k+=1

	cor_matrix = df_dcor.values.astype("float")
	sim_matrix = 1 - cor_matrix
	G = nx.from_numpy_matrix(sim_matrix)
	G = nx.relabel_nodes(G, lambda x: stocks[x])
	G.edges(data=True)
	H = G.copy()

	for (u, v, wt) in G.edges.data("weight"):
		if wt >= 1 - 0.325:
			H.remove_edge(u, v)
		if u == v:
			H.remove_edge(u, v)

	edges, weights = zip(*nx.get_edge_attributes(H, "weight").items())
	pos = nx.kamada_kawai_layout(H)

	with sns.axes_style("whitegrid"):
		plt.figure(figsize=(10, 5))
		plt.title("Distance correlation network of adjusted closing prices", size=13)
		deg = H.degree
		nodelist = []
		node_sizes = []

		for n, d in deg:
			nodelist.append(n)
			node_sizes.append(d)

		nx.draw_networkx_nodes(H, pos, node_color="#DA70D6", nodelist=nodelist, node_size=np.power(node_sizes, 2.33), alpha=0.8, linewidths=5)
		nx.draw_networkx_labels(H, pos)
		cmap = sns.cubehelix_palette(3, as_cmap=True, reverse=True)
		nx.draw_networkx_edges(H, pos, edgelist=edges, style="solid", edge_color=weights, edge_cmap=cmap, edge_vmin=min(weights), edge_vmax=max(weights))
		sm = plt.cm.ScalarMappable(cmap=cmap,  norm=plt.Normalize(vmin=min(weights),  vmax=max(weights)))
		sm._A = []
		plt.colorbar(sm)
		plt.axis("off")
		return plt


st.title("Optimal Asset Allocation")

symbol = load_data()
symbol_dict = dict(zip(symbol["Name"], symbol["Symbol"]))
symbol_inv_dict = {v: k for k, v in symbol_dict.items()}

st.sidebar.markdown("# Search by")
search = st.sidebar.radio("", ("Name", "Symbol"))

init_list_symbol = ["AAPL", "AMZN", "GOOG", "NFLX", "TSLA"]
init_list = ["Apple Inc.", "Amazon.com Inc.", "Alphabet Inc.", "Netflix Inc.", "Tesla Inc."]

st.markdown("### Select stocks")

random_select = st.radio("Select stocks at random", ("Yes", "No"), index=1)
if random_select == "Yes":
	num_col1, _ = st.columns(2)
	with num_col1:
		num_stocks = st.number_input("Select no. of stocks", min_value=2, max_value=30, value=5, key="num_stocks")

if search == "Symbol":
	if random_select == "Yes":
		init_list_symbol = random.sample(list(symbol_inv_dict.keys()), num_stocks)
	stocks = st.multiselect("", list(symbol["Symbol"]), init_list_symbol)
else:
	if random_select == "Yes":
		init_list = random.sample(list(symbol_dict.keys()), num_stocks)
	stocks = st.multiselect("", list(symbol["Name"]), init_list)
	stocks = [symbol_dict[i] for i in stocks]

if len(stocks) == 1:
	st.success("Please select stocks :)")
	st.stop()

st.sidebar.title("Select no. of years")
years = st.sidebar.slider(label="", min_value=1, max_value=5, value=3, step=1, key="year")

data = fetch_data(stocks, years*365)

nan_cols = list(set(data.columns) - set(data.dropna(axis=1).columns))
if len(nan_cols) > 0:
	st.write("Following stocks are dropped as they contain NaN values:")
	if search == "Symbol":
		st.write(nan_cols)
	else:
		st.write([symbol_inv_dict[i] for i in nan_cols])
	data = data.dropna(axis=1)
	stocks = list(data.columns)

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

max_sharpe = max_sharpe_ratio(mean_returns, log_returns, cov_matrix, risk_free_rate, sharpe_ratio_benchmark=1.0)
sdp, rp = portfolio_annualized_performance(max_sharpe["x"], mean_returns, cov_matrix)
max_sharpe_allocation = pd.DataFrame(max_sharpe["x"], index=data.columns, columns=["allocation"])
max_sharpe_allocation["allocation"] = [round(i*100, 2)for i in max_sharpe_allocation["allocation"]]
max_sharpe_allocation = max_sharpe_allocation.T

min_vol = min_variance(mean_returns, cov_matrix)
sdp_min, rp_min = portfolio_annualized_performance(min_vol["x"], mean_returns, cov_matrix)
min_vol_allocation = pd.DataFrame(min_vol.x,index=data.columns, columns=["allocation"])
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

if st.sidebar.checkbox("Distance Correlation Network", False):
	try:
		st.pyplot(correlation_network(data, stocks))
	except:
		st.write("NetworkX error :(")
		pass

begin_backtest = (date.today()-timedelta(365)).year
end_backtest = date.today().year

st.sidebar.title("Backtest on {} - {}".format(begin_backtest, end_backtest))
backtest = search = st.sidebar.radio("", ("Yes", "No"), index=1)
if backtest == "Yes":
	st.title("Backtest on {}-{}".format(begin_backtest, end_backtest))
	st.markdown("**Note**: Data used from **{} to {}** for caculation".format(date.today()-timedelta(365*(years+1)), date.today()-timedelta(365)))
	data_backtest = fetch_data(stocks, years*365, backtest=True)

	returns_backtest = data_backtest.pct_change()
	mean_returns_backtest = returns_backtest.mean()
	cov_matrix_backtest = returns_backtest.cov()
	risk_free_rate_backtest = 0.0178
	log_returns_backtest = np.log(data_backtest/data_backtest.shift(1))
	log_returns_backtest = log_returns_backtest.dropna()

	max_sharpe_backtest = max_sharpe_ratio(mean_returns_backtest, log_returns_backtest, cov_matrix_backtest, risk_free_rate_backtest, sharpe_ratio_benchmark=1.0)
	sdp_backtest, rp_backtest = portfolio_annualized_performance(max_sharpe_backtest["x"], mean_returns_backtest, cov_matrix_backtest)
	max_sharpe_allocation_backtest = pd.DataFrame(max_sharpe_backtest["x"], index=data_backtest.columns, columns=["allocation"])
	max_sharpe_allocation_backtest["allocation"] = [round(i*100, 2)for i in max_sharpe_allocation_backtest["allocation"]]
	max_sharpe_allocation_backtest = max_sharpe_allocation_backtest.T

	min_vol_backtest = min_variance(mean_returns_backtest, cov_matrix_backtest)
	sdp_min_backtest, rp_min_backtest = portfolio_annualized_performance(min_vol_backtest["x"], mean_returns_backtest, cov_matrix_backtest)
	min_vol_allocation_backtest = pd.DataFrame(min_vol_backtest.x,index=data_backtest.columns, columns=["allocation"])
	min_vol_allocation_backtest["allocation"] = [round(i*100,2)for i in min_vol_allocation_backtest["allocation"]]
	min_vol_allocation_backtest = min_vol_allocation_backtest.T

	an_vol_backtest = np.std(returns_backtest)*np.sqrt(252)
	an_rt_backtest = mean_returns_backtest*252

	st.write("### Maximum Sharpe Ratio Portfolio Allocation")
	st.write("Annualized Return:", round(rp_backtest, 2))
	st.write("Annualized Volatility:", round(sdp_backtest, 2))
	st.table(max_sharpe_allocation_backtest)

	st.write("### Minimum Volatility Portfolio Allocation")
	st.write("Annualized Return:", round(rp_min_backtest, 2))
	st.write("Annualized Volatility:", round(sdp_min_backtest, 2))
	st.table(min_vol_allocation_backtest)

	st.write("### Individual Stock Returns and Volatility")
	for i, txt in enumerate(data_backtest.columns):
		st.write("**{}:** Annualized Return: {}; Annualized Volatility: {}".format(txt, round(an_rt_backtest[i], 2), round(an_vol_backtest[i], 2)))

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

	df_test = pdr.get_data_yahoo(list(data_backtest.columns), start=date.today()-timedelta(365))["Adj Close"]
	ret_sharpe = 0
	for i in df_test.columns:
		ret_sharpe += df_test[i]*max_sharpe_allocation.reset_index(drop=True)[i][0]
	ret_sharpe /= 100
	ret_vol = 0
	for i in df_test.columns:
		ret_vol += df_test[i]*min_vol_allocation.reset_index(drop=True)[i][0]
	ret_vol /= 100
	df_test["Max Sharpe"] = ret_sharpe
	df_test["Min Volatility"] = ret_vol
	if df_test.isna().sum().sum() > 0:
		df_test = df_test.dropna()

	fig_backtest = go.Figure()
	fig_backtest.add_trace(go.Scatter(x=df_test.index, y=df_test["Max Sharpe"], name="Max Sharpe"))
	fig_backtest.add_trace(go.Scatter(x=df_test.index, y=df_test["Min Volatility"], name="Min Volatility"))
	fig_backtest.update_layout(yaxis_title="Price in $", title="<b>Portfolio performance in {}-{} based on aforementioned allocation</b>".format(begin_backtest, end_backtest))
	st.plotly_chart(fig_backtest)

	mspr = np.round((df_test["Max Sharpe"][-1] - df_test["Max Sharpe"][0])/df_test["Max Sharpe"][0] * 100, 2)
	mvpr = np.round((df_test["Min Volatility"][-1] - df_test["Min Volatility"][0])/df_test["Min Volatility"][0] * 100, 2)
	st.write("**Max Sharpe Portfolio Return ({} - {}):**".format(begin_backtest, end_backtest), mspr, "%")
	st.write("**Min Volatility Portfolio Return ({} - {}):**".format(begin_backtest, end_backtest), mvpr, "%")