"""
This file was made to make plotting ML results easier and cleaner
"""
import matplotlib.pyplot as plt

def plot_loss(history, val=False, ylim=None):
	plt.plot(history.history['loss'], label='loss')
	if val:
		plt.plot(history.history['val_loss'], label='val_loss')
	if ylim is not None:
		plt.ylim([0, 10])
	plt.xlabel('Epoch')
	plt.ylabel('Error [SOH]')
	plt.title('Loss Curve')
	plt.legend()
	plt.grid(True)


def plot_prediction(val_predict, val_labels, scatter=False, ylim=None, MG=False):

	x = [x for x in range(len(val_predict))]
	if scatter:
		plt.scatter(x,val_predict, label='prediction')
		plt.scatter(x,val_labels, label='actual')
	else:
		plt.plot(x,val_predict, label='prediction')
		plt.plot(x,val_labels, label='actual')
	plt.xlabel('Characterization Cycle', fontsize=16)
	plt.ylabel('SOH [%]', fontsize=16)
	if MG:
		plt.title('SOH Prediction Cell 14, 15', fontsize=16)
	else:
		plt.title('SOH Prediction Cell 2, 8, 11', fontsize=16)
	if ylim is not None:
		plt.ylim(ylim)
	plt.legend()
	plt.grid(True)

def calc_error(val_predict, val_labels):
	abs_error = []
	for idx, item in enumerate(val_predict):
		abs_error.append(abs(item - val_labels[idx]))

	mae = sum(abs_error)/len(abs_error)
	max_error = max(abs_error)
	return abs_error, mae, max_error

def plot_error(error, scatter=False):
	if scatter:
		plt.scatter(range(0, len(error)), error)
	else:
		plt.plot(error)
	plt.title('Prediction Error', fontsize=16)
	plt.xlabel('Characterization Cycle', fontsize=16)
	plt.ylabel('Abs. Error', fontsize=16)

def plot_prediction_and_error(val_predict, val_labels, scatter=False,
								ylim=None, figsize=(10,5), save=None,
								MG=False):
	abs_error, mae, max_error = calc_error(val_predict, val_labels)

	# plt.rcParams["figure.figsize"] = figsize
	fig = plt.figure(figsize=(figsize))
	plt.subplot(121)
	plot_prediction(val_predict, val_labels, scatter=scatter, ylim=ylim, MG=MG)

	plt.rcParams["figure.figsize"] = figsize
	plt.subplot(122)
	plot_error(abs_error)

	print(f'Mean absolute error: {mae}')
	print(f'Max error: {max_error}')

	if save is not None:
		fig.savefig('C:\\Users\\seanb\\OneDrive\\Documents\\PRIMED\\Images\\CNN\\'
			+ save
			+ '.png', dpi=400)

