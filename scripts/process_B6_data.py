"""
This file contains the code for processing battery data needed for ML models.

July 20, 2022

Sean Buchanan
"""

# third party imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

# local imports
from batlab.process import bat_io as bio


def load_data(batch_names):
	"""
	Load data from local files.

	Parameters
	----------
	batch_names: list -> batches in string format e.g. ["B6T10V0", "B6T11V0"].

	Returns
	-------
	batch_list: list -> batch objects describing battery data for a given batch.

	"""
	batch_list = []
	for batch in batch_names:
		batch_, shape_ = bio.load(batch)
		batch_list.append(batch_)

	return batch_list

def get_data_location(batch_list, s_ind, s_name, s_type):
	"""
	Get file location and row #'s of data in the file.

	Parameters
	----------
	batch_list: list -> batch objects describing battery data for a given batch.
	s_ind: int -> step index to select.
	s_name: string -> step name to select e.g. "Cc_crg"
	s_type: string -> step type to select e.g. "C" or "D"

	Returns
	-------
	step_data_info: dict -> contains tuples of file location row # pairs.
	"""

	step_data_info = {}
	for idx, batch in enumerate(batch_list):

		for cell in batch:
			c_num = cell["cell_num"]
			C = cell["d_c_rate"]
			DOD = cell["dod"]
			
			steps = np.array([step for step in cell if step["name"] == s_name
								  and step["type"] == s_type and step["s_ind"] == s_ind])
			
			temp = []
			for step in steps:
				temp.append((step["file"], step["rows"]))
				
			if c_num in step_data_info.keys():
				step_data_info[c_num] += temp
			else:
				step_data_info[c_num] = temp

	return step_data_info 	

def get_cc_charge_data(crg_step_data_info):
	"""
	Open data files and retrive relevant characterization CC charge data.
	Order data dataframes into dictionary that is cycle index by cell number.

	Parameters
	----------
	crg_step_data_info: dict -> contains tuples of file location row # pairs.

	Returns
	-------
	data_dict: dict -> key = cell number
					   value = list of dataframes containing constant current
							   charge data.
	"""

	cc_crg_data_list = []
	data_dict = {}
	prev_file = None

	print('Making data dictionary ', end='')
	for key, value in crg_step_data_info.items():
		for elem in value:

			# some cycles are split between two files
			# hence the list.
			if type(elem[0]) == list:
				temp1 = pd.read_csv(elem[0][0])
				temp2 = pd.read_csv(elem[0][1])

				temp_data1 = temp1[elem[1][0]-2:]
				temp_data2 = temp2[:elem[1][1]-2]

				temp_data2.index = temp_data2.index + elem[1][0]-2
				temp = pd.concat([temp_data1, temp_data2])
				cc_crg_data_list.append(temp)
				prev_file = elem[0][0][1]
				print('.', end='')
				continue
				
			if prev_file == elem[0]:
				cc_crg_data_list.append(temp[elem[1][0]-2:elem[1][1]-2])
			else:
				temp = pd.read_csv(elem[0])
				cc_crg_data_list.append(temp[elem[1][0]-2:elem[1][1]-2])
	
			prev_file = elem[0]
			print('.', end='')

		data_dict[key] = cc_crg_data_list
		cc_crg_data_list = []

	return data_dict	

def remove_cycles(data, cycles, output=False):
	"""
	Remove chosen cycles from the passed data dictionary.
	
	Parameters
	----------
	data: is a data dictionary.
	cycles: list -> contains tuples which are (cell, cycle_num) 
					indicating which cycles to remove form which cells.
	output: bool -> True for output data, False for input data.

	Returns
	-------
	data_dict_copy: dict -> a copy of the data dictionary with the specified cycles removed.

	"""

	data_dict_copy = copy.deepcopy(data)
	
	if output:
		for cell, cycle in cycles:
			for idx, elem in enumerate(data_dict_copy[cell]):
				if (cell, elem[0]) in cycles:
					del data_dict_copy[cell][idx]

	else:
		for cell, cycle in cycles:
			for idx, elem in enumerate(data_dict_copy[cell]):
				if (cell, elem['Cycle_Index'].iloc[0]) in cycles:
					del data_dict_copy[cell][idx]

	return data_dict_copy

def remove_cells(data, cells):
	"""
	Remove chosen cells from a data dictionary.

	Parameters
	----------
	data: dict -> data dictionary
	cells: list -> list of dictionary keys to be removed (cell #'s in this case)

	Returns:
	--------
	data_dict_copy: dict -> a copy of the data dictionary with the specified cells removed.

	"""

	data_dict_copy = copy.deepcopy(data)
	for key in cells:
		_ = data_dict_copy.pop(key)

	return data_dict_copy

def reduce_to_min_length(data_dict, initial_min=100000):
	"""
	Reduces the lengths of data in data_dict to the shortest one in the set.

	Parameters:
	-----------
	data_dict: dict -> key = cell number
					   value = list of dataframes containing constant current
							   charge data.
	initial_min: int -> some large # used as the initial minimum. Default is 1000000.

	Returns:
	--------
	data_dict_copy: dict -> a copy of data_dict with all DataFrames the same length
							as the shortest one.
	"""

	data_dict_copy = copy.deepcopy(data_dict)

	min = initial_min
	for key, value in data_dict.items():
		for elem in value:
			size = len(elem)
			if size < min:
				min = size

	for key, value in data_dict.items():
		for idx, elem in enumerate(value):
			data_dict_copy[key][idx] = elem.iloc[:min]

	return data_dict_copy


def split_data(data_dict, split=0.3, seed=3, print_keys=True):
	"""
	Split data into training and validation data sets.

	Parameters:
	----------
	data_dict: dict -> key = cell number, value = list of DataFrames of cell data
	split: float -> between 0 and 1. Represents the percentage of data that 
					will be used for validation.
	seed: int -> specifies the seed that random.seed() uses.
	print_keys: bool -> specifies if the selected key distribution is printed or not.

	Returns:
	-------
	train_data, val_data: tuple -> training and validation portions of the passed data.
	"""

	random.seed(seed)
	train_keys = sorted(random.sample( list(data_dict), int(len(data_dict)*(1-split)) ) )
	val_keys = sorted([ key for key in data_dict if key not in train_keys ])
	if print_keys:
		print(f'Training keys: {train_keys}')
		print(f'Validation keys: {val_keys}')

	# Split input data into training and validation datasets
	train_data = {}
	val_data = {}
	for key, value in data_dict.items():
		if key in train_keys:
			train_data[key] = value
		elif key in val_keys:
			val_data[key] = value

	return train_data, val_data

def rename_temp_col(data_dict_1):
	"""
	Takes a data dictionary and renames the temperature columns of its dataframes
	to Temp(C)_1 and Temp(C)_2. Obviously this is highly dependent on data format
	so be aware if used for data that isn't batch 6.

	Parameters:
	----------
	data_dict_1: dict -> dictionary of lists of DataFrame's

	Returns:
	-------
	data_dict: dict -> dictionary of lists of DataFrames with temperature col's renamed.
	"""
	
	data_dict = copy.deepcopy(data_dict_1)

	for key in data_dict:
		for elem in data_dict[key]:
			if 'Temp(C)_1' not in list(elem.columns):
				# rename temperature column
				col_names = list(elem.columns)
				for idx, name in enumerate(col_names):
					if name.endswith('_1'):
						col_names[idx] = 'Temp(C)_1'
					elif name.endswith('_2'):
						col_names[idx] = 'Temp(C)_2'
				elem.columns = col_names

	return data_dict

def concat_data_dict(data_dict_list):
	"""
	Concatenates data dictionaries together in the order of lowest index first.

	Parameters:
	-----------
	data_dict_list: list -> contains data dictionaries where the first element
							contains the earliest data and so on.
	"""

	data_dict = copy.deepcopy(data_dict_list[0])
	i = 1
	while i < len(data_dict_list):
		for key in data_dict:
			for idx, elem in enumerate(data_dict[key]):
				try:
					# check if idx exists, else continue. Differences will be caught later.
					data_dict_list[i][key][idx]
				except IndexError:
					continue

				data_dict[key][idx] = pd.concat([ elem, data_dict_list[i][key][idx] ])

		i += 1

	return data_dict

def reduce_to_min_length(data_dict, initial_min=100000):
	"""
	Reduces the lengths of data in data_dict to the shortest one in the set.

	Parameters:
	-----------
	data_dict: dict -> key = cell number
					   value = list of dataframes containing constant current
							   charge data.
	initial_min: int -> some large # used as the initial minimum. Default is 1000000.

	Returns:
	--------
	data_dict_copy: dict -> a copy of data_dict with all DataFrames the same length
							as the shortest one.
	"""

	data_dict_copy = copy.deepcopy(data_dict)

	min = initial_min
	for key, value in data_dict.items():
		for elem in value:
			size = len(elem)
			if size < min:
				min = size

	for key, value in data_dict.items():
		for idx, elem in enumerate(value):
			data_dict_copy[key][idx] = elem.iloc[:min]

	return data_dict_copy

def extend_data_dict(data_dict, initial_max=0):
	"""
	Extend the DataFrames in the data dictionary with the value in the last row
	such that it is the same length of the longest DataFrame in the dictionary.

	Also increments the index and time columns accordingly.

	Parameters:
	-----------
	data_dict: dict -> data dictionary
	initial_max: int -> initial maximum size. Default=0

	Returns:
	--------
	data_dict_copy: dict -> a copy of the passed data dict with all df's
							the same length as the longest.
	"""

	data_dict_copy = copy.deepcopy(data_dict)

	maxi = initial_max
	for key, value in data_dict.items():
		for elem in value:
			size = len(elem)
			if size > maxi:
				maxi = size

	for key, value in data_dict.items():
		for idx, elem in enumerate(value):
			n = maxi - len(elem)
			if n != 0:
				extension = elem.iloc[[-1]*n]
				extension.index = [ idx + step + 1 for step, idx in enumerate(extension.index) ]
				extension.loc[:,'Test_Time(s)'] = [ (time + idx*5) for idx, time in enumerate(extension.loc[:,'Test_Time(s)']) ]
				extension.loc[:,'Step_Time(s)'] = [ (time + idx*5) for idx, time in enumerate(extension.loc[:,'Step_Time(s)']) ]
				data_dict_copy[key][idx] = pd.concat( [elem, extension] )

	return data_dict_copy

def get_data_array(data_dict, headers):
	"""
	Takes in a dictionary of lists of dataframes and returns a numpy array of the columns of
	the specified headers. Also renames the temperature columns.

	NOTE: If the column naming could be generalized in a separte function
		  that would be a significant improvement.

	Parameters
	----------
	data_dict: dict -> dictionary of lists of DataFrame's
	headers: list -> strings of column names to be copied to the array

	Returns
	-------
	output_array: numpy.ndarray -> array of data from chosen DataFrame col's.
								   Of size (m x n x c) where m is the number
								   of cells.cycles and n is the number of timesteps
								   in the cycle and c is the number of columns that
								   were selected.
	"""
	output_array = []
	for key in data_dict:
		for elem in data_dict[key]:
			if 'Temp(C)_1' not in list(elem.columns):
				# rename temperature column
				col_names = list(elem.columns)
				for idx, name in enumerate(col_names):
					if name.endswith('_1'):
						col_names[idx] = 'Temp(C)_1'
					elif name.endswith('_2'):
						col_names[idx] = 'Temp(C)_2'
				elem.columns = col_names
				
			# if elem[headers].to_numpy().shape[-1] != len(headers):
			output_array.append(elem[headers].to_numpy())

	return np.array(output_array)

def correct_array_shape(arr, shape, axis=1):
	"""
	Append the last row of arr to arr until it has the same shape as shape.

	Parameters:
	-----------
	arr: numpy.ndarray -> data array
	shape: numpy.ndarray.shape -> desired shape of arr.

	Returns:
	--------
	arr_copy: numpy.ndarray -> arr of shape shape with extra rows equal
							   to arr[-1].
	"""

	arr_copy = arr.copy()
	temp = []
	num_rows = shape - arr.shape[1]
	if num_rows == 0:
		return arr
	else:
		for idx, elem in enumerate(arr):
			last_rows = [ elem[-1,:] for num in range(num_rows) ]
			temp.append( np.append(arr_copy[idx],last_rows,axis=0) )

	return np.array(temp)

def get_SOH(batch_list, datetime=False):
	"""
	Calculate SOH from battery data

	Parameters:
	----------
	batch_list: list -> contains batch objects previously loaded

	Returns:
	-------
	SOH_dict: dict -> key = cell number, value = list of (cycle #, SOH value)
	"""

	SOH_dict = {}

	for idx, batch in enumerate(batch_list):
		for cell in batch:
			c_num = cell["cell_num"]
			
				
			dis_steps = [step for step in cell if step["name"] == "Cc_dis"
								  and step["type"] == 'C' and step["s_ind"] == 10]


			SOHs = [(step["c_ind"], step["d_cap"]/4*100) for step in dis_steps]
			# SOHs = SOHs/SOHs[0]

			# SOH_dict[c_num] = SOHs

			if c_num in SOH_dict.keys():
				SOH_dict[c_num] += SOHs
			else:
				SOH_dict[c_num] = SOHs

	return SOH_dict

def remove_repeated_cycles(data_dict):
	"""

	"""

	for key in data_dict:
		for idx, elem in enumerate(data_dict[key]):
			if idx == 0:
				pass
			elif (elem['Cycle_Index'].iloc[0] == prev_idx):
				print(key, idx)
				data_dict[key].pop(idx-1);
			prev_idx = elem['Cycle_Index'].iloc[-1]

def remove_repeated_soh_cycles(data_dict):
	"""

	"""

	for key in data_dict:
		for idx, elem in enumerate(data_dict[key]):
			if idx == 0:
				pass
			elif (elem[0] == prev_idx):
				print(key, idx)
				data_dict[key].pop(idx-1);
			prev_idx = elem[0]



################## PLOTTING FUNCTIONS ##########################

def plot_data_dict(data_dict, channel, skip_list=[], step_time=True):
	"""
	Plot data from data dictionaries.

	NOTE: A subplot version of this would be better.

	Parameters:
	----------
	data_dict: dict -> data dictionary.
	channel: string -> specifies which column to plot.
	skip_list: list -> specifies which cycles to skip.
	step_time: bool -> specifies which time stamps to use so that the
					   plot looks correct.

	"""

	temp_bool = False
	if channel == 'Temp': temp_bool = True

	for key in data_dict:
		plt.figure(dpi= 100, facecolor='w', edgecolor='k')
		plt.grid(which="both", axis="both", color="C7", alpha=.5, zorder=1)

		for elem in data_dict[key]:

			if elem["Cycle_Index"].iloc[0] in skip_list:
				continue
			if temp_bool:
				channel = [ key for key in elem.keys() if key.endswith('_1')][0]
			if step_time:
				t = elem['Step_Time(s)']/60
			else:
				t = (elem['Test_Time(s)'] - elem['Test_Time(s)'].iloc[0])/60
			plt.plot(t, elem[channel], label=f'Cycle: {elem["Cycle_Index"].iloc[0]}')

		plt.title(f'Cell {key}')
		plt.xlabel('Time (minutes)')
		plt.ylabel(channel)
		_ = plt.legend()


def plot_array(arr, col, cycles=12, legend=True):
	"""
	Plot data arrays

	Parameters:
	----------
	arr: numpy.ndarray -> data array.
	cycles: int -> number of cycles that each cell have. Default=12. 
	"""
	count = 0
	plt.figure(dpi= 100, facecolor='w', edgecolor='k')
	for idx, elem in enumerate(arr):
		count += 1
		if idx % cycles == 0 and idx != 0:
			plt.title(f'Cell: {idx % count}')
			plt.legend()
			plt.figure(dpi= 100, facecolor='w', edgecolor='k')
			count = 1
		plt.plot(elem[:,col], label=f'cycle: {count}')
		
	plt.title(f'Cell: {count}')
	
	if legend:
		plt.legend()

def array_subplots(arr, rows, cols, titles=None, figsize=(10,8)):
	"""
	Plot subplots of data arrays.

	Parameters:
	----------
	arr: list or dict -> contains numpy.ndarray data arrays.
	rows: int -> number of subplot rows.
	cols: int -> number of subplot cols.
	titles: list -> contains title for the lot axis.
	"""

	fig, ax = plt.subplots(rows, cols, figsize=figsize)

	if type(arr) is dict:
		titles = [ key for key, value in arr.items() ]
		arr = [ value for key, value in arr.items() ]
	elif type(arr) is list:
		pass
	else:
		print('Input data is of wrong type. Needs to be list or dict.')
		return 0

	for k, arr2 in enumerate(arr):
		count = 0
		for idx, elem in enumerate(arr2):
			ax[k,0].plot(elem[:,0])
			ax[k,1].plot(elem[:,1])
			ax[k,0].set_title(titles[k])
			ax[k,1].set_title(titles[k])
			ax[k,0].set_ylabel('Voltage (V)')
			ax[k,1].set_ylabel('Current (A)')
	plt.tight_layout()