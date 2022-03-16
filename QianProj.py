from datetime import timedelta
import numpy as np
import pandas as pd
import argparse
import torch
import json
import os

from add_csv import csv_to_sqlite, csv_to_json
from sqlnet.dbengine import DBEngine
from sqlova.utils.utils_wikisql import *
from train import construct_hyper_param, get_models

#### prediction ####################

def get_args():
	parser = argparse.ArgumentParser()
	# parser.add_argument("--model_file", required=True, help='model file to use (e.g. model_best.pt)')
	# parser.add_argument("--bert_model_file", required=True, help='bert model file to use (e.g. model_bert_best.pt)')
	# parser.add_argument("--bert_path", required=True, help='path to bert files (bert_config*.json etc)')
	# parser.add_argument("--data_path", required=True, help='path to *.jsonl and *.db files')
	# parser.add_argument("--split", required=True, help='prefix of jsonl and db files (e.g. dev)')
	# parser.add_argument("--result_path", required=True, help='directory in which to place results')
	args = construct_hyper_param(parser)
	return args


def load_models(args):
	BERT_PT_PATH	= './data_and_model'
	path_model_bert	= './model_bert_best.pt'
	path_model		= './model_best.pt'
	model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, True, path_model_bert, path_model)
	return model, model_bert, tokenizer, bert_config


def my_get_fields(t, data_tables):
	### t: list of dict
	### data_tables: dict
	nlu, nlu_t, tb, hds = [], [], [], []
	for t1 in t:
		nlu.append(		t1['question'])
		nlu_t.append(	t1['question_tok'])
		tbid =			t1['table_id']
		tb.append(data_tables[tbid])
		hds.append(data_tables[tbid]['header'])
	return nlu, nlu_t, tb, hds


def my_predict( data_loader, data_table, 
				model, model_bert, bert_config, tokenizer,
				max_seq_length,
				num_target_layers, path_db, dset_name,
				EG=False, beam_size=4):
	
	model.eval()
	model_bert.eval()

	# engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
	engine = DBEngine(path_db)
	results = []
	for _, t in enumerate(data_loader):
		nlu, nlu_t, tb, hds = my_get_fields(t, data_table)
		wemb_n, wemb_h, l_n, l_hpu, l_hs, \
		nlu_tt, t_to_tt_idx, tt_to_t_idx \
			= get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
							num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
		if not EG:
			# No Execution guided decoding
			s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs)
			pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
			pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
			pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
		else:
			# Execution guided decoding
			prob_sca, prob_w, prob_wn_w, \
			pr_sc, pr_sa, pr_wn, pr_sql_i \
				= model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
									 l_hs, engine, tb,
									 nlu_t, nlu_tt,
									 tt_to_t_idx, nlu,
									 beam_size=beam_size)
			# sort and generate
			pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
			# Following variables are just for consistency with no-EG case.
			pr_wvi = None # not used
			pr_wv_str=None
			pr_wv_str_wp=None

		pr_sql_q = generate_sql_q(pr_sql_i, tb)

		for b, (pr_sql_i1, pr_sql_q1) in enumerate(zip(pr_sql_i, pr_sql_q)):
			results1 = {}
			results1["query"] = pr_sql_i1
			results1["table_id"] = tb[b]["id"]
			results1["nlu"] = nlu[b]
			results1["sql"] = pr_sql_q1
			results.append(results1)

	return results


#### deal with data ###################

## 不需要了
def read_csv_to_table(csv_path):
	# file_name as table_id
	table_id = csv_path.split('/')[-1][:-4]
	df = pd.read_csv(csv_path)
	headers = df.columns.tolist()
	rows = []
	for _, row in df.iterrows():
		rows.append(row.tolist())
	print(rows)


## TODO: add_csv
def create_table_and_db():
	pass


def read_scripts(txt_path):
	nlu = []
	with open(txt_path, 'r') as f:
		line = f.readline()
		while line:
			if line.endswith('\n'):
				nlu.append(line[:-1])
			else:
				nlu.append(line)
			line = f.readline()
	return nlu


## TODO: with tools in annotate_ws.py
def split_scripts(nlu):
	nlu_t = []
	for nlu1 in nlu:
		nlu_t.append(nlu1.split(' '))
	return nlu_t


def get_tables(tb_path):
	table = {}
	with open(tb_path) as f:
		for _, line in enumerate(f):
			t1 = json.loads(line.strip())
			table[t1['id']] = t1
	return table


def prepare_data():
	sc_paths	= [	'./Qian_data/company_script.txt', 
					'./Qian_data/product_script.txt',]
	sc_tableids	= [	'company_table',
					'product_table',]
	
	nlu		= []
	nlu_t	= []
	tbid	= []
	for i in range(len(sc_paths)):
		nlu_i	= read_scripts(sc_paths[i])
		nlu_t_i	= split_scripts(nlu_i)
		nlu.extend(nlu_i)
		nlu_t.extend(nlu_t_i)
		tbid.extend([sc_tableids[i]] * len(nlu_i))

	data = []
	for i in range(len(nlu)):
		data.append({
			'question':		nlu[i],
			'question_tok':	nlu_t[i],
			'table_id':		tbid[i],
		})
	
	return data
	

if __name__ == '__main__':

	dset_name = 'qian'
	save_path = './Qian_data/'

	### model
	args = get_args()
	model, model_bert, tokenizer, bert_config = load_models(args)
	
	### data
	db_path = './Qian_data/qian.db'
	tb_path = './Qian_data/qian.tables.jsonl'
	data_table = get_tables(tb_path)
	
	data = prepare_data()
	data_loader = torch.utils.data.DataLoader(
		batch_size=args.bS,
		dataset=data,
		shuffle=False,
		num_workers=1,
		collate_fn=lambda x: x  # now dictionary values are not merged!
	)

	### predict
	with torch.no_grad():
		results = my_predict(data_loader, 
							 data_table, 
							 model,
							 model_bert,
							 bert_config,
							 tokenizer,
							 max_seq_length=args.max_seq_length,
							 num_target_layers=args.num_target_layers,
							 path_db=db_path,
							 dset_name=dset_name,
							 EG=True,	#args.EG,
							 )

	# save results
	save_for_evaluation(save_path, results, dset_name)

