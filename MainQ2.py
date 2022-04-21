#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
from img_classification import make_predictions
from PIL import Image
import numpy as np

st.title("Classification with Deep Learning Model (MobileNetV2)")
st.header("Pokemon classification Example")
st.text("Upload a photo of either bulbasaur, charmander, charizard, mewtwo, or pikachu only")
st.text("This example is only for the 5 above-mentioned pokemons")
st.text("Any other pokemons uploaded might produce weird results, but you could try anyway")

uploaded_file = st.file_uploader("Choose a pokemon ...", type="jpg")

labels_k = {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}
labels = {0: 'bulbasaur', 1: 'charmander', 2: 'mewtwo', 3: 'pikachu', 4: 'squirtle'}

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Pokemon.', use_column_width=True)
    st.write("")
#    show_image(image)
#    st.write("")

    st.text("Classifying ... ... ...")
    
    # this gets the probability of the maximum score
    max_p = np.sort(np.max(make_predictions(image, 'pokemon_classifier_mobilenetModel.h5'),axis=0))[-1]

    # this gets the probability of the 2nd maximum score
    max_2p = np.sort(np.max(make_predictions(image, 'pokemon_classifier_mobilenetModel.h5'),axis=0))[-2]

    # this gets the ratio of the confidence level of the best probability to the best+2nd best combined    
    value = max_p/(max_p+max_2p)
    percent = "{:.2%}".format(value)
    
    #code1 = print('The prediction is: ',labels[np.argmax(make_predictions(image, 'pokemon_classifier_model.h5'),axis=1).tolist()[0]])
    st.text('The prediction is: ...')
    
    variable_output = labels[np.argmax(make_predictions(image, 'pokemon_classifier_mobilenetModel.h5'))]
    font_size = 50
    html_str = f"""
    <style>
    p.a {{
        font: bold {font_size}px Courier;
        color: red;
    }}
    </style>
    <p class="a">{variable_output}</p>
    """
    st.markdown(html_str, unsafe_allow_html=True)
    
    if value >= 0.5:
        #code2 = print('I am very confident of the result! \nThe degree of confidence is: {}'.format(percent))
        st.write('I am very confident of the result! \nThe degree of confidence is: {}'.format(percent))
    else:
        #code3 = print('I am somewhat confident of the result. \nThe degree of confidence is: {}'.format(percent))
        st.write('I am not very confident of the result. \nThe degree of confidence is: {}'.format(percent))
    
    st.text('------------------------------------------')
    value_match = {}
    for k, num in zip(labels_k, make_predictions(image, 'pokemon_classifier_mobilenetModel.h5')[0]):
        value_match[k] = num

    for k,v in value_match.items():
        #code4 = print('Prediction of {} with a {:.2%} confidence'.format(k,v))
        st.write('Prediction of {} with a {:.2%} probability'.format(k,v))
else:
    st.text("Nothing happened")
    
st.text('------------------------------------------')


# In[ ]:




