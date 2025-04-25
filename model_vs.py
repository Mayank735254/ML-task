#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Mayank735254/ML-task/blob/main/model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[2]:


import os
os.listdir()


# In[3]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('/content/description.txt')
data


# In[4]:


def load_data(file_path):
  with open(file_path,'r',encoding='utf-8') as f:
    data=f.readlines()
  data=[line.strip().split(' ::: ') for line in data ]
  return data


# In[5]:


train_data=load_data('/content/train_data.txt')
train_df=pd.DataFrame(train_data,columns=['ID','Title','Genre','Description'])

test_data=load_data('/content/test_data.txt')
test_df=pd.DataFrame(test_data,columns=['ID','Title','Description'])

test_solution=load_data('/content/test_data_solution.txt')
test_solution_df=pd.DataFrame(test_solution,columns=['ID','Title','Genre','Description'])


# In[6]:


print("Train Data:")
train_df



# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features=5000)


X_train_tfidf=vectorizer.fit_transform(train_df['Description'])
X_test_tfidf=vectorizer.transform(test_df['Description'])


print(f"Training data shape: {X_train_tfidf.shape}")
print(f"Test data shape: {X_test_tfidf.shape}")



# In[8]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
y_train=label_encoder.fit_transform(train_df['Genre'])
print(f"Unique genres in the training data: {label_encoder.classes_}")


# In[13]:


from sklearn.linear_model import LogisticRegression
linear_model=LogisticRegression(max_iter=500)
linear_model.fit(X_train_tfidf,y_train)

y_pred=linear_model.predict(X_test_tfidf)
Predicted_genres = label_encoder.inverse_transform(y_pred)

test_df['Predicted_Genre'] = Predicted_genres
test_df[['Title','Predicted_Genre']]



# In[15]:


test_df['Predicted_Genre'] = Predicted_genres

merged_df = pd.merge(test_solution_df[['ID', 'Genre']], test_df[['ID','Predicted_Genre']], on='ID')
merged_df


# In[16]:


from sklearn.metrics import accuracy_score,classification_report
accuracy=accuracy_score(merged_df['Genre'],merged_df['Predicted_Genre'])
print(f"Accuracy: {accuracy:4f}")


print("\nClassification Report:")
print(classification_report(merged_df['Genre'],merged_df['Predicted_Genre']))



# In[17]:


from sklearn.naive_bayes import MultinomialNB
nb_model=MultinomialNB()
nb_model.fit(X_train_tfidf,y_train)



# In[18]:


y_pred_nb=nb_model.predict(X_test_tfidf)
Predicted_genres_nb = label_encoder.inverse_transform(y_pred_nb)


test_df['Predicted_Genre_NB'] = Predicted_genres_nb
merged_df_nb = pd.merge(test_solution_df,test_df[['ID','Predicted_Genre_NB']], on = 'ID')



# In[19]:


from sklearn.metrics import accuracy_score,classification_report

accuracy_nb=accuracy_score(merged_df_nb['Genre'],merged_df_nb['Predicted_Genre_NB'])
print(f"Naive Bayes Accuracy : {accuracy_nb}")


print("Naive Bayes Classification Report:")
print(classification_report(merged_df_nb['Genre'],merged_df_nb['Predicted_Genre_NB']))


# In[20]:


from sklearn.svm import SVC
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_tfidf, y_train)


# In[21]:


y_pred_svm = svm_model.predict(X_test_tfidf)
Predicted_genres_svm = label_encoder.inverse_transform(y_pred_svm)


test_df['Predicted_Genre_SVM'] = Predicted_genres_svm
merged_df_svm = pd.merge(test_solution_df, test_df[['ID', 'Predicted_Genre_SVM']], on='ID')



# In[22]:


from sklearn.metrics import accuracy_score, classification_report

accuracy_svm = accuracy_score(merged_df_svm['Genre'], merged_df_svm['Predicted_Genre_SVM'])
print(f"SVM Accuracy: {accuracy_svm}")


print("SVM Classification Report:")
print(classification_report(merged_df_svm['Genre'], merged_df_svm['Predicted_Genre_SVM'],target_names=label_encoder.classes_))



# In[29]:


zoner_Description = [
    'Explosive fight scenes in the city streets',
    'A haunted mansion that traps its visitors',
    'A brave adventurer in search of lost treasure',
    'A forbidden romance in the 1920s',
    'A daring rescue mission with a love interest'

    ]

test_data_tfidf = vectorizer.transform(zoner_Description)


y_pred_lr = linear_model.predict(test_data_tfidf)
predicted_genres_lr = label_encoder.inverse_transform(y_pred_lr)

y_pred_nb = nb_model.predict(test_data_tfidf)
predicted_genres_nb = label_encoder.inverse_transform(y_pred_nb)

y_pred_svm = svm_model.predict(test_data_tfidf)
predicted_genres_svm = label_encoder.inverse_transform(y_pred_svm)


print("Predicted Genres using Logistic Regression :", predicted_genres_lr)
print("Predicted Genres using Naive Bayes :", predicted_genres_nb)
print("Predicted Genres using SVM :", predicted_genres_svm)
print()
for i,message in enumerate(zoner_Description):
  print(f"Story : {message}")
  print(f"Status :\tNaive Bayes Prediction :{predicted_genres_nb[i]}")
  print(f"\t\tLogistic Regression Prediction :{predicted_genres_lr[i]}")
  print(f"\t\tSVM Prediction :{predicted_genres_svm[i]}")
  print("="*100)








# In[ ]:




