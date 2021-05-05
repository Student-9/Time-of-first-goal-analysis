import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chisquare
import matplotlib
from lifelines import KaplanMeierFitter

location = input('Please insert path of dataframe ')
location = location.strip('\"')

def fetch_data(path): 

	df = pd.read_csv(path, sep=';')
	score_data = df['Total Time'].tolist()
	scoreless_data = df['Scoreless Time'].tolist()
	Tees = df['T']
	Eves = df['E']

	while -1 in score_data: 
		score_data.remove(-1) #Removes -1's from total times

	scoreless_data = [x for x in scoreless_data if x == x] #Removes NaN's from durations of scoreless matches

	return score_data, scoreless_data, Tees, Eves

all_data = fetch_data(location)
first_goal_times = all_data[0]
scoreless_times = all_data[1]
T = all_data[2]
E = all_data[3]
M = max(first_goal_times)

total_sum = np.sum(first_goal_times) + np.sum(scoreless_times) #Total sum of first goal times and durations of scoreless matches
matches_with_goal = len(first_goal_times) #Number of matches in which a goal was scored
total_matches = len(first_goal_times) + len(scoreless_times)

def exponential_mle(x, sum_t=total_sum, n1=matches_with_goal): #Maximum likelihood estimation under the exponential distribution assumption

	lnL = x*(n1 - sum_t) + n1*np.log(1-np.exp(-x)) #log-likelihood
	return -lnL #We want to maximize the log-likelihood, which is equivalent to minimizing the negative log-likelihood, hence the (-) sign.


res = minimize(exponential_mle, x0=0.02, method ='Nelder-Mead',tol=10**(-10)) #Maximizes the log-likelihood function
mu_ml = res.x[0] #Maximum likelihood estimate

b = list(range(0,100,10)) #Bins
b.append(M)

def exp_freqs(m, n=total_matches, bins=b): #Calculates expected frequencies for some given mu
	
	exp_frequencies = []

	for i in range(0,10):
		r = np.exp(-m*bins[i])-np.exp(-m*bins[i+1])
		r = r*n
		exp_frequencies.append(r) 

	return exp_frequencies

expected_frequencies = exp_freqs(mu_ml)

def obs_freqs(data=first_goal_times):

	obs_frequencies = np.histogram(data, 
    	bins=10,
    	range=(1,101),
    	density=False) #Calculates observed frequencies

	return obs_frequencies[0]

observed_frequencies = obs_freqs()

chi_sq_result = chisquare(observed_frequencies, expected_frequencies, ddof=1) #Performs a chi-squared goodness of fit test
p_value = chi_sq_result.pvalue #p-value of chi-squared test

print(f'ML estimate: {mu_ml}')
print(f'p-value: {p_value}')

def min_chi_sq(x,o=observed_frequencies, N=total_matches, bins=b): #Minimum chi-squared estimation

	s = 0
	for i in range(0,10):

		s += (o[i]-N*(np.exp(-x*bins[i])-np.exp(-x*bins[i+1])))**2/(N*(np.exp(-x*bins[i])-np.exp(-x*bins[i+1])))

	return s

res_2 = minimize(min_chi_sq, x0=0.02, tol=10**(-10), options={'maxiter':10000}, method='Nelder-Mead') #Minimizes the chi-squared statistic
mu_min_chi_sq = res_2.x[0] #Mimimum chi-squared estimate

expected_frequencies_2 = exp_freqs(mu_min_chi_sq) 
new_result = chisquare(observed_frequencies, expected_frequencies_2, ddof=1)
p_value_2 = new_result.pvalue

print(f"\nMin chi squared estimate: {mu_min_chi_sq}") 
print(f'p-value: {p_value_2}') 

def plot_histogram(est_method=None, expected=[],  Bins=b, observed=observed_frequencies):

	if expected:
		labels = ['1-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','90+']

		x = np.arange(len(labels))
		y = list(range(0,100,10))
		width = 0.35

		fig, ax = plt.subplots()
		rects1 = ax.bar(x - width/2, observed, width, label='Observed Frequencies')
		rects2 = ax.bar(x + width/2, expected, width, label='Expected Frequencies')

		ax.set_ylabel('Frequencies')
		ax.set_title(f'Expected vs Observed Frequencies: {est_method}')
		ax.set_xticks(x)
		ax.set_yticks(y)
		ax.set_xticklabels(labels)
		ax.legend()

		fig.tight_layout()

		ax.bar_label(rects2, padding=2)

		plt.show()
	else:
		labels = ['1-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','90+']

		x = np.arange(len(labels))
		y = list(range(0,90,10))
		width = 0.35

		fig, ax = plt.subplots()
		rects = ax.bar(x, observed, width)

		ax.set_ylabel('Observed Frequencies')
		ax.set_xticks(x)
		ax.set_yticks(y)
		ax.set_xticklabels(labels)

		fig.tight_layout()

		ax.bar_label(rects, padding=2)

		plt.show()


plot_0 = plot_histogram('Maximum likelihood', expected_frequencies) #Plots observed vs expected frequencies under the ML method
plot_1 = plot_histogram('Minimum chi-squared', expected_frequencies_2) #Plots observed vs expected frequencies under the min X^2 method
plot_2 = plot_histogram() #Plots just the observed frequencies

def Product_limit_estimator(Tau=T,Eps=E): #Plots KM survival curve
	kmf = KaplanMeierFitter()
	kmf.fit(Tau, Eps)
	kmf.survival_function_
	kmf.cumulative_density_
	plt.xticks(range(0,110,10))
	plt.yticks(np.arange(0,1.1,0.1))
	med = kmf.median_survival_time_
	kmf.plot()
	plt.show()
	print(f'Median: {med}')

Product_limit_estimator()




















