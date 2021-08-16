
# coding: utf-8

# In[1]:


import graphlab


# In[ ]:


data=graphlab.SFrame("C:\Users\kumar\Downloads\chennaidata (4).csv")


# In[3]:


data


# In[ ]:





# # Exploring the data

# In[4]:


data.show()


# In[5]:


data['square_ft']


# # Simple regression model with square feet and price

# In[6]:


train_data,test_data=data.random_split(0.8,seed=0)


# In[7]:


train_data


# In[8]:


test_data


# # Bulid the regression model

# In[9]:


sqft_model=graphlab.linear_regression.create(train_data,target='price',features=["square_ft"])


# In[10]:


sqft_model


# # Evaluate the model

# In[11]:


sqft_model.evaluate(test_data)


# # lets see our predictions

# In[12]:


import matplotlib.pyplot as plt
plt.plot(test_data['square_ft'],test_data['price'],'*',
        test_data['square_ft'],sqft_model.predict(test_data),'-')
plt.show()


# # Exploring other feautures

# In[13]:


my_feautures=["bedrooms","bathrooms","square_ft","area","status"]


# In[14]:


data[my_feautures].show()


# # Building  regression model with more features

# In[15]:


my_feautures_model=graphlab.linear_regression.create(train_data,target='price',features=my_feautures)


# In[16]:


my_feautures_model


# In[17]:


print sqft_model.evaluate(test_data)
print my_feautures_model.evaluate(test_data)


# #  Apply learned models to predict house prices

# In[18]:


house1=data[data['square_ft']==1247]


# In[19]:


house1


# In[20]:


house1['price']


# In[21]:


sqft_model.predict(house1)


# In[22]:


my_feautures_model.predict(house1)


# In[23]:


roms_house={
    "square_ft":[2000],
    "place":["annanagar"],
    "bedrooms":[2]
}


# In[24]:


sqft_model.predict(test_data)


# In[25]:


my_feautures_model.predict(test_data)


# In[ ]:


"""
Sholinganallur
Perumbakkam
Poonamallee
Karayanchavadi
Nungambakkam
Kattankulathur
Kodambakkam
Royapettai
Egmore
Guduvancheri
Annanagar
Chetpet
Mogappair
Koyambedu
maduravoyil
Tambarameast
Tambaramwest
Avadi
Chromepet
Velachery
Teynampet
Adyar
Pallavaram
ECR
Thiruvanmiyur
Porur
Iyyanpanthangal
Kattankulathur
Mogappair
Alwarpet
Chetpet
Vanagaram
Tnagar
Saidapet
Ambattur
Vadapalani
Perungalathur
Vandalur
Ayanavaram
Choolaimedu
Kundrathur
Tiruvottiyur
Westmambalam
Tondiarpet
Korattur
Perambur
Pallikaranai
Madipakkam
Kolathur
Kodungaiyur
Moolakadai
Ponneri
Minjur
Nemilicheri
Thiruneermalai
Kandigai
Thirumudivakkam
Villivakkam
Vyasarpadi
Varadharajapuram
Kanathur
Manalinewtown
Ponniammanmedu
Pozhichalur
Ramavaram
Rajakilpakkam
Pattabiram
Perumbakkam
Selaiyur
120
pammal
chitlapakkam
nanmangalam
madambakkam
Padi
Mangadu
Ayapakkam
Annanur
sithalapakkam
anakaputhur
Valasaravakkam
Thiruverkadu
Jamalia
Puzhal
Madhavaram
Oragadam
"""


# # final predictions on houses given by user

# In[30]:


square_ft=int(raw_input("enter sqft \t"))
area=raw_input("ENter area\t")
bedrooms=int(raw_input("enter no of bedrooms\t"))
roms_house={
    "square_ft":[square_ft],
    "place":[area],
    "bedrooms":[bedrooms]
}
print sqft_model.predict(graphlab.SFrame(roms_house))
print my_feautures_model.predict(graphlab.SFrame(roms_house))


# In[ ]:





# In[ ]:





# In[ ]:




