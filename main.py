import os       # sslerror 발생 시 requests==2.27.1로 맞추고 line 2, 3 추가하면 됨
os.environ['CURL_CA_BUNDLE']=''

import re
import time
import pickle

import pandas as pd
from PIL import Image

from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import plotly.express as px

# set variable
filename = 'multi_col_606.png'
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

for i in reversed(range(1, num_col)):
    r = result_list[i]
    temp = pd.DataFrame(r.split(' | ')).T
    df = pd.concat([df, temp], axis=0)

df.columns = col_row
df = df.reset_index(drop=True)

# 값에 unit이 같이 들어있는지 체크
a = df.iat[0, 1]
num_a = re.sub(r"[^.0-9]", "", a)
new_a = re.sub(r"[.0-9]", "", a)
print(f'{num_a}, {new_a}')



for c in col_row[1:]:
    df = df.astype({c: 'float'})

# plotly
fig = px.line(df, x=x_label, y=y_label, markers=True)
fig.update_xaxes(rangeslider_visible=True)
fig.show()