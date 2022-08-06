#!/usr/bin/env python
# coding: utf-8

# # FakeNewsDetection

# In this notebook, we will be solving a use case of 'Fake News Detection' - Predict if a news published in an article is 'Real' or 'Fake' for a sample data using ML Algorithms!

# **Note**: Update variables under **Variables** section if required before running the notebook. To run notebook cell by cell, click on a cell and click **Run** button below the Menu bar. Or to run all cells, select **Cell --> Run** All from Menu bar.

# ### Variables

# In[2]:


#Specify the input filename
TRAINFILE=r"fake_news_train.csv"
TESTFILE=r"fake_news_test.csv"
#Specify the ratio of the data to subset for prediction
test_data_ratio = 0.05

#By default, EXPOSE_AS_API is False. 
#If it is True then this kit will be exposed as a rest API, and it can be consumed through URL http://127.0.0.1:5000/predict
EXPOSE_AS_API=False

#By default, TRAIN_MODEL is False and it uses pretrained model(fakenewsmodel.pkl). 
#If TRAIN_MODEL is True then it uses the training data to build new model which will be used for the prediction.
TRAIN_MODEL=False  


# ### Import libraries to detect fake news

# In[3]:


from detect import FakeNewsDetection
from app import FakeNewsApiService


# ### Training

# In[3]:


fakenews = FakeNewsDetection(TRAINFILE, test_data_ratio)


# In[4]:


if TRAIN_MODEL:
    fakenews.train_model()


# ###  Prediction

# In[5]:


fakenews.test_news(TESTFILE)


# # FakeNewsDetection API Service
# The following code exposes this solution as a rest API. This feature can be turn on by setting the variable EXPOSE_AS_API=True. Input and output details along with the endpoint URL details are given below.

# #### Prediction API url
# POST http://127.0.0.1:5000/predict 

# #### API input
# | Field | Description | Example |
# | :- | :- | :- |
# | news_text | News text from the article | "BGMI not Banned In India; Here's What Google And Krafton Said" |
# 
# Example json 
# ```
# { 
#     "news_text": "BGMI not Banned In India; Here's What Google And Krafton Said"
# }
# ```

# #### API output
# | Field | Description | Example |
# |:-- | :-- | :-- |
# | news_text | News text from the article | "BGMI not Banned In India; Here's What Google And Krafton Said" |
# |label | If the value is 'fake' it is Fake news.If the value is 'real' it is real news |"real"|
# |probability | Confidence level of prediction |"0.79"|
# 
# Example json
# ```
# {
#     "label": "fake",
#     "news_text": "BGMI not Banned In India; Here's What Google And Krafton Said",
#     "probability": 0.79
# }
# ```

# In[5]:


if EXPOSE_AS_API:
    api=FakeNewsApiService()
    api.start()


# This is a starter notebook for FakeNewsDetection using 'LogisticRegressionCV' model. More detailed code can be found in the **'FakeNewsDetection-analysis.ipynb'** notebook in the current directory.

# In[ ]:




