#%%
import json
#%%
data = []
path_sql = './data_and_model/test_tok.jsonl'
with open(path_sql) as f:
    for idx, line in enumerate(f):
        t1 = json.loads(line.strip())
        data.append(t1)

#%%
table = {}
path_table = './data_and_model/test.tables.jsonl'
with open(path_table) as f:
    for idx, line in enumerate(f):
        t1 = json.loads(line.strip())
        table[t1['id']] = t1
# %%
