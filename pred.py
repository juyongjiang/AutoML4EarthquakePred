import pandas as pd
import numpy as np
import os
import zipfile
import warnings
warnings.filterwarnings('ignore')
import argparse
import csv
from numpy.lib.function_base import extract
from preprocess import generate_features
from area_map import area_groups, magn_level
from utils import *
import skopt

def get_range_data(time_range):
	if not os.path.exists(args.save_path): os.makedirs(args.save_path)
	file_name = 'EM&GA_' + time_range[0] + '-' + time_range[1] # time1 - time2
	extractpath = os.path.join(args.save_path, file_name) # save unzip files
	if not os.path.exists(extractpath):
		frzip = zipfile.ZipFile(os.path.join(args.zip_path, file_name+'.zip'), 'r')
		frzip.extractall(extractpath)
		frzip.close()
	else:
		print("Already unzip. Skip ...")
	
	# check whether there exists .zip files
	em_path = os.path.join(f'{extractpath}', f'EM_{time_range[0]}-{time_range[1]}')
	ga_path = os.path.join(f'{extractpath}', f'GA_{time_range[0]}-{time_range[1]}')
	print(em_path, ga_path)
	em_id, ga_id = set(), set()
	for filename in os.listdir(em_path):
		if(filename.endswith('.csv')):
			em_id.add(eval(filename.split('_')[0]))
			continue
		with zipfile.ZipFile(em_path+filename, 'r') as frzip:
			frzip.extractall(em_path)
	for filename in os.listdir(ga_path):
		if(filename.endswith('.csv')):
			ga_id.add(eval(filename.split('_')[0]))
			continue
		with zipfile.ZipFile(ga_path+filename, 'r') as frzip:
			frzip.extractall(ga_path)

	return em_path, em_id, ga_path, ga_id

def write_csv(args, save_path, longitude=-1, latitude=-1, max_mag=-1):
	pre_Range_left = stamp2string(string2stamp(time_range[1]) + 86400)
	pre_Range_right = stamp2string(string2stamp(time_range[1]) + 86400*7)
	# write earthquake results into csv
	if not os.path.exists(save_path):
		with open(save_path, mode='a', newline='') as predict_file:
			csv_writer = csv.writer(predict_file)
			csv_writer.writerow(['whether', 'longitude', 'latitude', 'magnitude', 'starttime', 'endtime'])
			if max_mag == -1:
				csv_writer.writerow([0, '', '', '', pre_Range_left, pre_Range_right])
			else:
				csv_writer.writerow([1, longitude, latitude, max_mag, pre_Range_left, pre_Range_right])
	else:
		with open(save_path, mode='a', newline='') as predict_file:
			csv_writer = csv.writer(predict_file)
			if max_mag == -1:
				csv_writer.writerow([0, '', '', '', pre_Range_left, pre_Range_right])
			else:
				csv_writer.writerow([1, longitude, latitude, max_mag, pre_Range_left, pre_Range_right])


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='arguments')
	parser.add_argument('--time_range', default=['20220529', '20220604'], nargs="+")
	parser.add_argument('--zip_path', default='./dataset/AETA_20220529-20220702/', type=str) # or './dataset/AETA_20220501-20220521/'
	parser.add_argument('--save_path', default='./dataset/data_week/', type=str)
	parser.add_argument('--model_save_path', default='./saved', type=str)
	parser.add_argument('--prediction', default='./prediction.csv', type=str)
	parser.add_argument('--eq_gt', default='./ground_truth.csv', type=str)
	args = parser.parse_args()
	
	# time_range_list = [['20220501', '20220507'], ['20220508', '20220514'],] # current week data to predict next week 
	time_range_list = [['20220529', '20220604'], ['20220605', '20220611'], ['20220612', '20220618']]
	
	# model_dict = {0:'svm', 1:'svm', 2:'svm', 3:'decision_trees', 4:'mlp', 5:'svm', 6:'svm', 7:'mlp'}
	# model_dict = {0:'svm', 1:'mlp', 2:'svm', 3:'decision_trees', 4:'mlp', 5:'decision_trees', 6:'svm', 7:'decision_trees'}
	model_dict = {0:'naive_bayes', 1:'mlp', 2:'decision_trees', 3:'decision_trees', 4:'mlp', 5:'decision_trees', 6:'decision_trees', 7:'decision_trees'}
	if os.path.exists(args.prediction): os.remove(args.prediction)
	if os.path.exists(args.eq_gt): os.remove(args.eq_gt)
	print('Submission Recreate!')

	for time_range in time_range_list:
		print("==>time range", time_range)
		em_path, em_id, ga_path, ga_id = get_range_data(time_range) # obtain the corresponding em&ga data in time range
		print(em_path, em_id, '\n', ga_path, ga_id)

		max_mag, eq_area = -1, -1 # the max magnitude in all areas, eq_area is the index of each area
		for area in (0, 1, 2, 3, 4, 5, 6, 7):
			print("==>", area)
			## concat all EM and GA data from all stations of each area in a week range
			sid = area_groups[area]['id'] & em_id & ga_id # get station id in each area
			if len(sid)==0: continue
			em_list = []
			for id_num in sid:
				em_list.append(pd.read_csv(os.path.join(em_path, f'{id_num}_magn.csv')))
			em_data = pd.concat(em_list)
			del em_list
			ga_list = []
			for id_num in sid:
				ga_list.append(pd.read_csv(os.path.join(ga_path, f'{id_num}_sound.csv')))
			ga_data = pd.concat(ga_list)
			del ga_list # release memory 

			## get representative features by weekly time interval
			start_stamp = string2stamp(time_range[0])
			em_data = generate_features(em_data, 7, 'magn', start_stamp)
			ga_data = generate_features(ga_data, 7, 'sound', start_stamp)
			_final_res = pd.merge(em_data, ga_data, on='Day', how='left')
			_final_res.fillna(0, inplace=True)
			_final_res.drop('Day', axis=1, inplace=True)
			features = _final_res.iloc[-1] # use the last week's features to predict the next week's earthquake
			
			features = np.array(features.values).reshape(1, -1)
			ml_model = skopt.load(os.path.join(args.model_save_path, f'{area}_{model_dict[area]}_model.joblib'))
			print("Loding ml model: ", os.path.join(args.model_save_path, f'{area}_{model_dict[area]}_model.joblib'))
			# print(features.shape)

			## machine learning model prediction
			preds = ml_model.predict(features)[0] # return index
			if preds == 0:
				# print('preds=0')
				continue
			else:
				if(preds > max_mag):
					max_mag = preds
					eq_area = area

		if max_mag != -1:
			# use the center of area represents the next possible earthquake location
			longitude = (area_groups[eq_area]['range'][2] + area_groups[eq_area]['range'][3]) / 2
			latitude = (area_groups[eq_area]['range'][0] + area_groups[eq_area]['range'][1]) / 2
			print(f'magnitude:{magn_level[max_mag]}, longitude:{longitude}, latitude:{latitude}')
			write_csv(args, args.prediction, longitude, latitude, magn_level[max_mag])
		else:
			write_csv(args, args.prediction, max_mag=-1)

		## print the ground truth in the next week
		pre_Range_left = string2stamp(time_range[1]) + 86400
		pre_Range_right = string2stamp(time_range[1]) + 86400*7
		groud_truth = pd.read_csv(os.path.join(args.zip_path, f'EC_{stamp2string(string2stamp(time_range[1]) + 86400)}-{stamp2string(string2stamp(time_range[1]) + 86400*7)}.csv'))
		_eq_gt = groud_truth[(groud_truth['Timestamp']<pre_Range_right) & (groud_truth['Timestamp']>=pre_Range_left)]
		if len(_eq_gt) == 0:
			print("eq_gt", len(_eq_gt), _eq_gt)
			write_csv(args, args.eq_gt, max_mag=-1)
		elif len(_eq_gt) == 1: # have the once earthquake
			print("eq_gt", len(_eq_gt), _eq_gt)
			print("=========================", _eq_gt['Magnitude'])
			write_csv(args, args.eq_gt, _eq_gt['Longitude'].values[0], _eq_gt['Latitude'].values[0], _eq_gt['Magnitude'].values[0])
		else:
			print("eq_gt", len(_eq_gt), _eq_gt)
			_eq_max = _eq_gt.iloc[_eq_gt['Magnitude'].argmax()]
			gt_max_mag = _eq_max['Magnitude']
			gt_longitude = _eq_max['Longitude']
			gt_latitude = _eq_max['Latitude']
			write_csv(args, args.eq_gt, gt_longitude, gt_latitude, gt_max_mag)
