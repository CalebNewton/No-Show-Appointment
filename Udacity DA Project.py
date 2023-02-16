#!/usr/bin/env python
# coding: utf-8

# # Obtaining our dataset

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('darkgrid')


# This dataset collects information from 100000 medical appointments in brazil and is focused on the question of whether or not patients show up for their appointment. This dataset was originally sourced from </https://www.kaggle.com/>

# In[3]:


#Loading our dataset

df = pd.read_csv("C:/Users/Caleb Henry/Desktop/Udacity/noshowappointments-kagglev2-may-2016.csv")


# In[4]:


#lets see top 5 samples of our data

df.head(5)


# In[5]:


#Lets obtain more information about this dataset

df.info()


# In[6]:


#Lets obtain statistical info of our dataset. Transposing, we obtain the following values for each features.

df.describe().T


# # Data Wrangling

# From the result obtained using .info(), we can see the datatype of each of our features. We noticed some features possess the wrong datatype which has to be changed later. Hence, the need for data wrangling
# 
# Data wrangling involves removing erros and combining complex datasets to make them more accessible and easier to analyze. Common Problems to look out for before analysis can be done are as follows; Check for
# 
# incorrect datatypes
# missing data
# duplicates
# structural problems like different column names
# mismatch number of records;

# In[7]:


#Checking for missing or null values

df.isnull().sum().all()


# In[8]:


#Checking for duplicates

df.duplicated().sum()


# In[9]:


#Checking the Age features

df.Age.value_counts()


# In[10]:


#Categorizing Neighbourhood

df.Neighbourhood.value_counts()


# In[11]:


#Categorizing Handicap

df.Handcap.value_counts()


# In[12]:


#From the above result, we know in reality that no person age can be -1, hence, lets locate the sample with age -1

df[df['Age'] == -1]


# In[13]:


#replace the -1 age with 0, since its impossible for age to be -1

df.Age.replace({-1:0},inplace=True)


# In[14]:


#AppointmentID wont make an significance in our prediction and so, we should remove this

df.drop(['AppointmentID'],axis=1,inplace=True)


# In[15]:


df.head()


# In[16]:


#Checking for duplicates again. Though the dataset shows duplicates after the appointmentid was removed but this cant be removed cos these data seems to be very important

df.duplicated().sum()


# In[17]:


#Renaming the columns

df.rename(columns={'Hipertension':'Hypertension', 'Handcap': 'Handicap', 'No-show': 'No_show'},inplace=True)


# #### Solving datatype issues for each features

# In[18]:


#Converting scholarship, hypertension, diabetes, alcoholism and sms_received to booleans

for b in ['Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'SMS_received']:
    df[b] = df[b].astype('bool')


# In[19]:


#Converting ScheduledDay and AppointmentDay to datetime dtype

for b in ['ScheduledDay', 'AppointmentDay']:
    df[b] = pd.to_datetime(df[b])


# In[20]:


#Converting Gender, Neighbourhood, handicap

for b in ['Gender', 'Neighbourhood', 'Handicap']:
    df[b] = df[b].astype('category')


# From the no_show column, we know when the response is NO, it means the patient made it for the appointment but when it's YES, it means the patient didnt make it.

# In[21]:


#Replacing NO and YES with 0 and 1 inorder to easily convert the datatype

df.No_show.replace({'No': 1, 'Yes': 0},inplace=True)


# In[22]:


df.No_show = df.No_show.astype('bool')


# In[23]:


df.dtypes


# In[24]:


#Checking our No_show feature

df.No_show.value_counts()


# Questions to solve;
# 
# 1. What's the age of men that showed up the most for their appointment?
# 2. What's the proportion of male to female who showed up for their appointment?
# 3. What's the relationship between (age and gender), (gender and No_show), (age, gender and No_show)?
# 4. What's the number of males with hypertension that showed up for their appointment?

# # Exploratory Data Analysis (EDA)

# No_show

# In[25]:


#The number of people that did show up for an appointment. No_show is True i.e 1, if the patient showed up and False i.e 0, if the patient didnt show up

df[df['No_show'] == 1].No_show.count()


# In[26]:


#The number of people that didnt show up for an appointment

df[df['No_show'] == 0].No_show.count()


# AppointmentDay

# In[27]:


#The day with the lowest attendance

df.AppointmentDay.min()


# In[28]:


#The number of patients on the above day

df.AppointmentDay.value_counts().min()


# In[29]:


#The day with the highest attendance

df.AppointmentDay.max()


# In[30]:


#The number of patients on the above day

df.AppointmentDay.value_counts().max()


# Gender

# In[31]:


#Total values of male and female patients

total_gender = df.groupby('Gender').count().Age
total_gender


# Neighbourhood

# In[32]:


df.Neighbourhood.value_counts()


# In[33]:


df.Neighbourhood.value_counts().plot(kind='bar',title='Neighbourhood and number of patients',alpha=1)
plt.xlabel('Neighbourhood', fontsize=18)
plt.ylabel('Number of patients', fontsize=18)


# Hypertension

# In[34]:


df.Hypertension.value_counts()


# Relationships between two variables;

# In[35]:


#Age and patients that showed up

df.groupby('Age').No_show.value_counts()


# In[36]:


#Females and their age range

df.query('Gender == "F"').Age.value_counts()


# In[37]:


df.query('Gender == "F"').Age.max()


# In[38]:


#males and their age range

df.query('Gender == "M"').Age.value_counts()


# In[39]:


df.query('Gender == "M"').Age.max()


# In[40]:


#Find the number of males and females that showed up for their appointment i.e Gender and No_show

No_show_by_gender = df.groupby(['Gender', 'No_show'])['Gender'].count()
No_show_by_gender


# In[41]:


No_show_by_gender.describe(include='all')


# In[42]:


#Calculate frequencies of No_show for females

No_show_femaleProp = No_show_by_gender['F']/total_gender['F']
No_show_femaleProp


# In[43]:


#Calculate frequencies of No_show for males

No_show_maleProp = No_show_by_gender['M']/total_gender['M']
No_show_maleProp


# In[44]:


#Gender and Neighbourhood

df.groupby('Gender').Neighbourhood.value_counts()


# In[45]:


#Neighbourhood with their patients that were awarded a scholarship

df.query('Scholarship == True').Neighbourhood.value_counts()


# In[46]:


#Neighbourhoods with their patients that showed up

df.query('No_show == True').Neighbourhood.value_counts()


# In[47]:


#Statisical rep of the relationship between Neighbourhood and No_show

df.query('No_show == True').Neighbourhood.value_counts().describe(include='all')


# In[48]:


#Correlation between features i.e variables

df.corr('pearson')


# Relationship between more than two variables;

# In[49]:


#Ages of females that showed up for their appointment

df.query('Gender == "F" and No_show == True').Age.value_counts()


# In[50]:


#Ages of males that showed up for their appointment

df.query('Gender == "M" and No_show == True').Age.value_counts()


# In[51]:


#Ages of males that didnt show up for their appointment

df.query('Gender == "M" and No_show == False').Age.value_counts()


# In[52]:


#Ages of females that didnt show up for their appointment

df.query('Gender == "F" and No_show == False').Age.value_counts()


# In[53]:


#The neighbourhood with scholarships that showed up for appointment

df.query('Scholarship == True and No_show == True').Neighbourhood.value_counts()


# In[54]:


#The neighbourhood without scholarship that showed up for appointment

df.query('Scholarship == False and No_show == True').Neighbourhood.value_counts()


# In[55]:


#The neighbourhood without scholarships that didnt show up for appointment

df.query('Scholarship == False and No_show == False').Neighbourhood.value_counts()


# In[56]:


#The neighbourhood without scholarships that didnt show up for appointment

df.query('Scholarship == True and No_show == False').Neighbourhood.value_counts()


# We can see from above, the relationships that exist between those with or without scholarship that either attended or missed their appointments for each neighbourhood.

# In[57]:


#Females with scholarship that are handicapped

df.query('Gender == "F" and Scholarship == True and Handicap == True').count().Gender


# In[58]:


#Males with scholarship that are handicapped

df.query('Gender == "M" and Scholarship == True and Handicap == True').count().Gender


# In[59]:


#Males with hypertension that showed up for appointment

df.query('Gender == "M" and Hypertension == True and No_show == True').count().Gender


# In[60]:


#Females with hypertension that showed up for appointment

df.query('Gender == "F" and Hypertension == True and No_show == True').count().Gender


# In[61]:


#Ages of males that showed up for their appointment

df.query('Gender == "M" and No_show == True').Age.value_counts()


# In[62]:


#Males and Females with hypertension and diabetes that showed up for their appointments

df.query('Hypertension == True and Diabetes == True and No_show == True').Gender.value_counts()


# In[63]:


#Males and Females with hypertension and diabetes that showed up for their appointments after receiving sms notification

df.query('Hypertension == True and Diabetes == True and SMS_received == True and No_show == True').Gender.value_counts()


# In[64]:


#Females that are handicapped in a particular neighbourhood
df.query('Gender == "F" and Handicap == True').Neighbourhood.value_counts()


# In[65]:


#Males that are handicapped in a particular neighbourhood
df.query('Gender == "M" and Handicap == True').Neighbourhood.value_counts()


# # Data Visualisation

# In[66]:


def countp(col_name):
    sns.countplot(col_name)


# In[67]:


#Plot showing Ages

countp(df.Age);


# In[68]:


#Plot of Scholarship

countp(df.Scholarship);


# In[69]:


#Plot of Gender distribution

countp(df.Gender);


# In[70]:


#Plot for hypertension patients

countp(df.Hypertension)


# In[71]:


#The relationship between gender and No_show. this shows us the gender with a higher chance of showing up

sns.countplot(df.Gender,hue=df['No_show']);


# In[72]:


#plot of age against gender

sns.countplot(df.Age,hue=df['Gender']);


# In[73]:


sns.countplot(df.Neighbourhood, hue=df['No_show']);


# In[74]:


#Appointment distribution

sns.countplot(df['No_show']);


# In[75]:


df.groupby(['AppointmentDay', 'No_show'])['Gender'].count().hist();


# In[76]:


df.query('Hypertension == True and Diabetes == True and SMS_received == True and No_show == True').Gender.value_counts().plot(kind='bar', title='Graph of hypertensive and diabetic patients that received a sms and showed up');


# In[77]:


df.query('Gender == "M" and Handicap == True').Neighbourhood.value_counts().hist();


# # Conclusion

# From the above analysis and visualisation, the following can be observed based on questions asked;
# 
# * Men that were 0-yo, showed up the most for their appointments. They were about 1498
# * 80% men showed up for their appointments as against 79.6% women that showed up for their appointments. The % difference amongst gender can be said to be negligible since it's just about 0.4%
# * The oldest females were 115yo while the oldest men were 100yo. Hence, it can be seen they were more younger men than women as can be seen from the youngest males and females.
# * 57246 females out of 71840 showed up for their appointment while about 30962 males out of 38687 showed up for their appointments.
# * 1403 females of 0-yo showed up for their appointment while 1498 males of same age showed up for their appointment. It can be seen the lower the age, the higher their chances of making it to their appointment since less than 12 females of age 98 and above were able to show up while less than 13 males of age 95 and above did same.
# * 5347 males with hypertension showed up for their appointment

# # References

# </https://stackoverflow.com/>
# </https://janamalesova.github.io/>
