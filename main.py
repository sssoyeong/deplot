import os       # sslerror 발생 시 requests==2.27.1로 맞추고 line 2, 3 추가하면 됨
os.environ['CURL_CA_BUNDLE']=''

import re
import time
import pickle

import numpy as np
import pandas as pd
from PIL import Image

from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import plotly.express as px
import plotly.graph_objects as go


# set variable
filename = 'multi_col_21130.png'    # line 4개
# filename = 'multi_col_606.png'
# filename = 'two_col_102644.png'

# load pre-trained model
processor = Pix2StructProcessor.from_pretrained('google/deplot')
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

# load image
image = Image.open(filename)
image.show()

# # predict
# inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
# start_time = time.time()
# predictions = model.generate(**inputs, max_new_tokens=512)
# elapsed_time = time.time()-start_time

# print(processor.decode(predictions[0], skip_special_tokens=True))
# print(f'elapsed time: {elapsed_time:.4f} secs')

# result = processor.decode(predictions[0], skip_special_tokens=True)
# print(result)

# # save variables
# with open(f'result_{filename[:-4]}.pkl', 'wb') as f:
#     pickle.dump(result, f)

# load variables
with open(f'result_{filename[:-4]}.pkl', 'rb') as f:
    result = pickle.load(f)
print(result)

# convert string result to dataframe
x_list, y_list = [], []
result_list = result.split(' <0x0A> ')

num_col = result_list[0].count('|') + 1
num_row = len(result_list)

col_row = result_list[0].split(' | ')
df = pd.DataFrame()

for i in reversed(range(1, num_row)):
    r = result_list[i]
    # temp = pd.DataFrame(r.split(' | ')).T
    temp = r.split(' | ')
    temp_sub = [temp[0]] + [re.sub(r'[^ .0-9]', '', x) for x in temp[1:]]
    temp_sub = pd.DataFrame(temp_sub).T
    df = pd.concat([df, temp_sub], axis=0)

df.columns = col_row
df = df.reset_index(drop=True)
df[col_row[1:]] = df[col_row[1:]].apply(pd.to_numeric)
idx_original = df[col_row[0]]

# check unit
check_ex = result_list[1].split(' | ')[1]
unit = re.sub(r'[ .0-9]', '', check_ex)
exist_unit = False if unit == '' else True

# make traces between nodes
idx_trace = np.linspace(0, df.shape[0]-1, (df.shape[0]-1) * 1000 + 1)
list_trace = np.reshape(idx_trace, (len(idx_trace), 1))

for c in range(1, len(col_row)):
    temp = np.interp(idx_trace, df.index, df[col_row[c]])
    temp = np.reshape(temp, (len(temp), 1))
    list_trace = np.concatenate((list_trace, temp), axis=1)

df_trace = pd.DataFrame(list_trace, columns=col_row)

# plotly scatter with go
slct_cm = px.colors.qualitative.Vivid
fig = go.Figure()

for c in range(len(df.columns[1:])):
    col = df.columns[1:][c]
    fig.add_trace(
        go.Scatter(x=df.index, y=df[col], name=col, marker_color=slct_cm[c])
    )

for c in range(len(df.columns[1:])):
    col = df.columns[1:][c]
    fig.add_trace(
        go.Scatter(x=df_trace[col_row[0]], y=df_trace[col], name=col, opacity=0, showlegend=False,  marker_color=slct_cm[c])
    )
    print(col)

fig.update_layout(
    xaxis_title=col_row[0],
    xaxis = dict(
        tickmode = 'array',
        tickvals = df.index,
        ticktext = idx_original.values
    )
)
if df.shape[1] == 2:
    fig.update_layout(yaxis_title=col_row[1])
fig.show()



# try with px
import plotly.express as px
fig = px.line(df, x=df.columns[0], y=df.columns[1:], markers=True, color_discrete_sequence=px.colors.qualitative.Vivid)
fig.update_traces(mode="markers+lines")
fig.show()

