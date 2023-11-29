#####################################################
# DePlot Web Service v0.2
#####################################################
#
# (history)
#
# for v0.3: More test, support various types of ploting ...
#
# DePlot Web Service v0.2 by Jeongcheol Lee 23.11.28
#  - add basic except handling
# DePlot Web Service v0.1 by Jeongcheol Lee 23.11.27
#  - real-time linkage between dataframe and chart (modificable)
# DePlot Inference Code by Soyoung Park
#
#####################################################



import pygwalker as pyg
import pandas as pd
import streamlit.components.v1 as components
import streamlit as st

############################################################ 
# Streamlit 페이지의 너비 조정
st.set_page_config(
    initial_sidebar_state="collapsed",
    #page_title="Ex-stream-ly Cool App",
    #page_icon="🧊",
    layout="wide",
    #menu_items={
    #    'Get Help': 'https://www.extremelycoolapp.com/help',
    #    'Report a bug': "https://www.extremelycoolapp.com/bug",
    #    'About': "# This is a header. This is an *extremely* cool app!"
    #}
)
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

##########################################################
#from streamlit_oauth import OAuth2Component
#from dotenv import load_dotenv
#import os, sys, inspect
#load_dotenv()
#AUTHORIZE_URL = os.environ.get("AUTHORIZE_URL")
##TOKEN_URL = os.environ.get("TOKEN_URL")
#TOKEN_URL = "http://150.183.247.248:8180/auth/realms/EDISON2/protocol/openid-connect/token"
#REFRESH_TOKEN_URL = TOKEN_URL
#REVOKE_TOKEN_URL = TOKEN_URL
#CLIENT_ID = os.environ.get("CLIENT_ID")
#CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
#REDIRECT_URI_REST = os.path.basename(inspect.getfile(inspect.currentframe()))
#REDIRECT_URI_REST = REDIRECT_URI_REST.split(".")[0]
#REDIRECT_URI = os.environ.get("REDIRECT_URI")
#REDIRECT_URI = os.path.join(REDIRECT_URI, REDIRECT_URI_REST)
#SCOPE = os.environ.get("SCOPE")
## Create OAuth2Component instance
##AUTHORIZE_URL = "http://150.183.247.248:8180/auth/realms/EDISON2/protocol/openid-connect/auth"
##TOKEN_URL = "http://150.183.247.248:8180/auth/realms/EDISON2/protocol/openid-connect/token"
##REFRESH_TOKEN_URL = TOKEN_URL
##REVOKE_TOKEN_URL = TOKEN_URL
##CLIENT_ID = "edison-streamlit"
##CLIENT_SECRET = "00JmOkU6xS3B7kqSVWfyB31yszpLuMzM"
##REDIRECT_URI =  "https://dev248.edison.re.kr:9443/app/darwin"
##SCOPE = "openid"
##st.write(REDIRECT_URI)
#
#
## Create OAuth2Component instance
##AUTHORIZE_URL = "https://auth-test.edison.re.kr/realms/master/protocol/openid-connect/auth"
##TOKEN_URL = "https://auth-test.edison.re.kr/realms/master/protocol/openid-connect/token"
##REFRESH_TOKEN_URL = TOKEN_URL
##REVOKE_TOKEN_URL = TOKEN_URL
##CLIENT_ID = "streamlit"
##CLIENT_SECRET = "yW5HvK1CtTYcNbBfaEqN0ZPIOB1PRrF8"
##REDIRECT_URI =  "https://dev248.edison.re.kr:9443/app/pyg"
##SCOPE = "openid"
##st.write(REDIRECT_URI)
#
#
#
#
#oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, REFRESH_TOKEN_URL, REVOKE_TOKEN_URL)
#
#
##run_flag = False
## Check if token exists in session state
#if 'token' not in st.session_state:
#    # If not, show authorize button
#    run_flag = False
#    result = oauth2.authorize_button("Sign in with keycloak", REDIRECT_URI, SCOPE)
#    if result and 'token' in result:
#        # If authorization successful, save token in session state
#        st.session_state.token = result.get('token')
#        st.experimental_rerun()
#else:
#    # If token exists in session state, show the token
#    token = st.session_state['token']
#    run_flag = True
#    #if st.button("Refresh Token"):
#    #    # If refresh token button is clicked, refresh the token
#    #    token = oauth2.refresh_token(token)
#    #    st.session_state.token = token
#    #    st.experimental_rerun()# AxiosError: Request failed with status code 403
run_flag = True
##############################################################

import os
import time
import pandas as pd
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import plotly.express as px

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def getfig(df):
    x_label, y_label = df.columns[0], df.columns[1]
    fig = px.line(df, x=x_label, y=y_label, markers=True)
    fig.update_xaxes(rangeslider_visible=True)
    return fig

def deplot_inference(uploaded_file, device='cpu'):
    image = Image.open(uploaded_file)
    # set variable
    #filename = 'two_col_102644.png'
    if device=='gpu':
        device="cuda:0"# if torch.cuda.is_available() else "cpu"
    # load pre-trained model
    processor = Pix2StructProcessor.from_pretrained('google/deplot', cache_dir="./deplot")
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot',cache_dir="./deplot").to(device)
    # model size: about 3G

    # predict
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt").to(device)
    start_time = time.time()
    predictions = model.generate(**inputs, max_new_tokens=512)
    elapsed_time = time.time()-start_time

    print(processor.decode(predictions[0], skip_special_tokens=True))
    print(f'elapsed time: {elapsed_time:.4f} secs')

    result = processor.decode(predictions[0], skip_special_tokens=True)
    print(result)
    try:
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
    except:
        st.markdown("[ERR] failed to parsing the plot..")
        st.markdown("**Parsed Result by DePlot Model**")
        st.write(result)
        st.markdown('**Converted Table (modificable)**')
        edited_df = st.data_editor(df, num_rows="dynamic", on_change=save_edits)
        csv = convert_df(edited_df)
        st.download_button(
               "Press to Download as CSV",
               csv,
               os.path.basename(current_filename).replace(".","_")+".csv",
               "text/csv",
               key='download-csv'
            )
        if st.button("Make a Plot"):
            with st.spinner("Loading..."):
                st.plotly_chart(getfig(edited_df))
    # plotly
    #fig = px.line(df, x=x_label, y=y_label, markers=True)
    #fig.update_xaxes(rangeslider_visible=True)
    return df

def save_edits():
    st.session_state.count+=1

##################################
if run_flag == True:
    # 제목 추가
    st.title("DePlot: One-shot visual language reasoning by plot-to-table translation")
    st.image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/deplot_architecture.png")
    st.markdown("ArXiv link ( <https://arxiv.org/abs/2212.10505> )")
    st.markdown("**Citation**")
    st.markdown("If you want to cite this work, please consider citing the original paper:") 
    st.code('''@misc{liu2022deplot,
      title={DePlot: One-shot visual language reasoning by plot-to-table translation},
      author={Liu, Fangyu and Eisenschlos, Julian Martin and Piccinno, Francesco and Krichene, Syrine and Pang, Chenxi and Lee, Kenton and Joshi, Mandar and Chen, Wenhu and Collier, Nigel and Altun, Yasemin},
      year={2022},
      eprint={2212.10505},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}''')
    
    if "loaded" not in st.session_state:
        st.session_state.loaded = ""
        st.session_state.prev_filename = ""
        st.session_state.count = 0
    
    uploaded_file = st.file_uploader("Choose a plot image file (png, jpg, ..)")
    if uploaded_file is not None:
        st.markdown("**Original Image**")
        st.session_state.loaded = uploaded_file
        st.session_state.fromwhere = "uploaded"
    else:
        if st.button("Load SAMPLE Dataset"):
            with st.spinner("Loading..."):
                uploaded_file = "app/pages/deplot/testpng.png"
                st.markdown("**Original Image**")
                st.session_state.loaded = uploaded_file
                st.session_state.fromwhere = "sample"
    if "loaded" in st.session_state:
        if st.session_state.loaded != "":
            if st.session_state.fromwhere == "uploaded":    # from uploader
                current_filename = st.session_state.loaded.name
            elif st.session_state.fromwhere == "sample":    # from sample file
                current_filename = st.session_state.loaded
            st.image(st.session_state.loaded)    
            #
            if current_filename != st.session_state.prev_filename:
                df = deplot_inference(st.session_state.loaded, device='gpu')
                fig = getfig(df)
                st.session_state.prev_filename = current_filename
                st.session_state.fig = fig
                st.session_state.df = df
            st.markdown('**Converted Table (modificable)**')
            df = st.session_state.df
            edited_df = st.data_editor(df, num_rows="dynamic", on_change=save_edits)
            #st.markdown("**count**")
            #st.write(st.session_state.count)
            st.markdown('**DePlotted Chart**')
            st.plotly_chart(getfig(edited_df))
            csv = convert_df(edited_df)
            st.download_button(
               "Press to Download as CSV",
               csv,
               os.path.basename(current_filename).replace(".","_")+".csv",
               "text/csv",
               key='download-csv'
            )
