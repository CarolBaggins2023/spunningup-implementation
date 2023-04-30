import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_data(
		data,
		xaxis='Epoch',
		value="AverageEpisode_Reward",
		condition="Condition1",
		smooth=1,
		**kwargs
):
	if smooth > 1:
		"""
		smooth data with moving window average. that is,
			smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
		where the "smooth" param is width of that window (2k+1)
		"""
		y = np.ones(smooth)
		for datum in data:
			x = np.asarray(datum[value])
			z = np.ones(len(x))
			smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
			datum[value] = smoothed_x
	
	if isinstance(data, list):
		data = pd.concat(data, ignore_index=True)
	sns.set(style="darkgrid", font_scale=1.5)
	# sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
	sns.lineplot(data=data, x=xaxis, y=value, errorbar='sd', **kwargs)
	"""
	If you upgrade to any version of Seaborn greater than 0.8.1, switch from
	tsplot to lineplot replacing L29 with:
	
		sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

	Changes the colorscheme and the default legend style, though.
	"""
	plt.legend(
		loc='upper center', ncol=6, handlelength=1,
		mode="expand", borderaxespad=0., prop={'size': 13}
	)
	# plt.legend(loc='best').set_draggable(True)
	# plt.legend(loc='upper center', ncol=3, handlelength=1,
	#           borderaxespad=0., prop={'size': 13})
	
	"""
	For the version of the legend used in the Spinning Up benchmarking page,
	swap L38 with:

	plt.legend(loc='upper center', ncol=6, handlelength=1,
		mode="expand", borderaxespad=0., prop={'size': 13})
	"""
	
	xscale = np.max(np.asarray(data[xaxis])) > 5e3
	if xscale:
		# Just some formatting niceness: x-axis scale in scientific notation if max x is large
		plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
	
	plt.tight_layout(pad=0.5)


def get_datasets(logdir, condition=None):
	"""
	Recursively look through logdir for output files produced by spinup.logx.Logger.

	Assumes that any file "progress.txt" is a valid hit.
	"""
	global exp_idx
	global units
	datasets = []
	for root, _, files in os.walk(logdir):
		if 'progress.txt' in files:
			exp_name = None
			try:
				config_path = open(os.path.join(root, 'config.json'))
				config = json.load(config_path)
				if 'exp_name' in config:
					exp_name = config['exp_name']
			except:
				print('No file named config.json')
			condition1 = condition or exp_name or 'exp'
			condition2 = condition1 + '-' + str(exp_idx)
			exp_idx += 1
			if condition1 not in units:
				units[condition1] = 0
			unit = units[condition1]
			units[condition1] += 1
			
			try:
				exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
			except:
				print('Could not read from %s' % os.path.join(root, 'progress.txt'))
				continue
			performance = 'AverageTest_Episode_Reward' \
				if 'AverageTest_Episode_Reward' in exp_data else 'AverageEpisode_Reward'
			exp_data.insert(len(exp_data.columns), 'Unit', unit)
			exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
			exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
			exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
			datasets.append(exp_data)
	return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
	"""
	For every entry in all_logdirs,
		1) check if the entry is a real directory and if it is, pull data from it;
		2) if not, check to see if the entry is a prefix for a
		real directory, and pull data from that.
	"""
	logdirs = []
	for logdir in all_logdirs:
		if osp.isdir(logdir) and logdir[-1] == os.sep:
			logdirs += [logdir]
		else:
			basedir = osp.dirname(logdir)
			fulldir = lambda x: osp.join(basedir, x)
			prefix = logdir.split(os.sep)[-1]
			listdir = os.listdir(basedir)
			logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
	
	"""
	Enforce selection rules, which check logdirs for certain substrings.
	Makes it easier to look at graphs from particular ablations, if you
	launch many jobs at once with similar names.
	"""
	if select is not None:
		logdirs = [log for log in logdirs if all(x in log for x in select)]
	if exclude is not None:
		logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]
	
	# Verify logdirs
	print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
	for logdir in logdirs:
		print(logdir)
	print('\n' + '=' * DIV_LINE_WIDTH)
	
	# Make sure the legend is compatible with the logdirs
	assert not legend or (len(legend) == len(logdirs)), \
		"Must give a legend title for each set of experiments."
	
	# Load data from logdirs
	data = []
	if legend:
		for log, leg in zip(logdirs, legend):
			data += get_datasets(log, leg)
	else:
		for log in logdirs:
			data += get_datasets(log)
	return data


def make_plots(
		all_logdirs, legend=None, xaxis=None, values=None, count=False,
		font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean'
):
	data = get_all_datasets(all_logdirs, legend, select, exclude)
	values = values if isinstance(values, list) else [values]
	condition = 'Condition2' if count else 'Condition1'
	estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
	for value in values:
		plt.figure()
		plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator)
	plt.show()
