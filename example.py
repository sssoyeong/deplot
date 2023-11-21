# sslerror 발생 시 requests==2.27.1로 맞추고 아래 두 줄 추가하면 됨
import os
os.environ['CURL_CA_BUNDLE']=''

import time

from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

processor = Pix2StructProcessor.from_pretrained('google/deplot')
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
# https://github.com/vis-nlp/ChartQA/blob/4878877860d6c51dd95df9376cd9baa776e81068/ChartQA%20Dataset/val/png/5090.png
image = Image.open(requests.get(url, stream=True).raw)
image.show()

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
start_time = time.time()
predictions = model.generate(**inputs, max_new_tokens=512)
elapsed_time = time.time()-start_time

print(processor.decode(predictions[0], skip_special_tokens=True))
print(f'elapsed time: {elapsed_time:.4f} secs')

result = processor.decode(predictions[0], skip_special_tokens=True)
result2 = result.replace('<0x0A>', '\n')
print(result2)


url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/two_col_43.png"
image = Image.open(requests.get(url, stream=True).raw)
image.show()

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
start_time = time.time()
predictions = model.generate(**inputs, max_new_tokens=512)
elapsed_time = time.time()-start_time

print(processor.decode(predictions[0], skip_special_tokens=True))
print(f'elapsed time: {elapsed_time:.4f} secs')

result = processor.decode(predictions[0], skip_special_tokens=True)
result2 = result.replace('<0x0A>', '\n')
print(result2)
