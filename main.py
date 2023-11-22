import os       # sslerror 발생 시 requests==2.27.1로 맞추고 line 2, 3 추가하면 됨
os.environ['CURL_CA_BUNDLE']=''

import time
import pickle
import requests

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

# predict
inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
start_time = time.time()
predictions = model.generate(**inputs, max_new_tokens=512)
elapsed_time = time.time()-start_time

print(processor.decode(predictions[0], skip_special_tokens=True))
print(f'elapsed time: {elapsed_time:.4f} secs')

result = processor.decode(predictions[0], skip_special_tokens=True)
result2 = result.replace('<0x0A>', '\n')
print(result2)

# save variables
with open(f'result_{filename[:-4]}.pkl', 'wb') as f:
    pickle.dump(result, f)
    pickle.dump(result2, f)

# load variables
with open(f'result_{filename[:-4]}.pkl', 'rb') as f:
    result1 = pickle.load(f)
    result2 = pickle.load(f)

# convert string result to dataframe
x_list, y_list = [], []
result_list = result1.split(' <0x0A> ')
for r in result_list:
    x, y = r.split(' | ')
    x_list.append(x)
    y_list.append(y)

x_label = x_list.pop(0)
y_label = y_list.pop(0)
x_list.reverse()
y_list.reverse()
df = pd.DataFrame({x_label: x_list, y_label: y_list})
df = df.astype({y_label: 'float'})

# plotly
fig = px.line(df, x=x_label, y=y_label, markers=True)
fig.update_xaxes(rangeslider_visible=True)
fig.show()