import os
os.environ['CURL_CA_BUNDLE']=''

import time

import pandas as pd
from PIL import Image

from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import plotly.express as px


if __name__=='__main__':
    # set variable
    filename = 'two_col_102644.png'

    # load pre-trained model
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

    # load image
    image = Image.open(filename)

    # predict
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
    start_time = time.time()
    predictions = model.generate(**inputs, max_new_tokens=512)
    elapsed_time = time.time()-start_time

    print(processor.decode(predictions[0], skip_special_tokens=True))
    print(f'elapsed time: {elapsed_time:.4f} secs')

    result = processor.decode(predictions[0], skip_special_tokens=True)
    print(result)

    # convert string result to dataframe
    x_list, y_list = [], []
    result_list = result.split(' <0x0A> ')
    num_col = result_list[0].count('|') + 1
    num_row = len(result_list)

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