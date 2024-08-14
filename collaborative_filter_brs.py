
# DATA COLLECTION:-
import numpy as np
import pandas as pd

books = pd.read_csv('books.csv',low_memory=False)
users = pd.read_csv('users.csv',low_memory=False)
ratings = pd.read_csv('ratings.csv',low_memory=False)

#Humne data ko input kr liya h ab hum is data ma FEATURE ENGINEERING and DATA ANALYSIS karegay.
# FEATURE ENGINEERING:- ismme hum data ke andar koi mistake to nahi ye dekhte h and then un mistakes ko sahi krte h.(isse hamara modal or jada accurate outpute dega) 

# DATA ANALYSIS:- isme hum data analysis krke pettern analysis krte h.(iski help se hum wahi dataset banate h joki MACHINE LEARNING MODAL ke andar hum daaleygay,data analysis ma is data se direct report prepare krte h joki past and present ki information to dedeta h but future ki nahi.)
# (1) ratings.csv and books.csv ko merge krdiya and us file ka name rating_with_book_name h.
rating_with_books_name = ratings.merge(books,on='ISBN')
#----------------------------------------------------------------------------------------------------------------------------------------------
#collaborative_filter_brs
#
x=rating_with_books_name.groupby('User-ID').count()['Book-Rating']>200

user_jinhone_200se_jada_rating_ki_h_not_dfform=x[x].index


user_jinhone_200se_jada_rating_ki_h_in_dfform=rating_with_books_name[rating_with_books_name['User-ID'].isin(
user_jinhone_200se_jada_rating_ki_h_not_dfform)]



####books jinko 50 se jada rating mil h
y=user_jinhone_200se_jada_rating_ki_h_in_dfform.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index

final_ratings=user_jinhone_200se_jada_rating_ki_h_in_dfform[user_jinhone_200se_jada_rating_ki_h_in_dfform['Book-Title'].isin(famous_books)]
pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt.fillna(0,inplace=True)

from sklearn.metrics.pairwise import cosine_similarity
similarity_score=cosine_similarity(pt)


def recommend(book_name):
    #index fetch kr rahe h
    index =np.where(pt.index==book_name)[0][0]
    similer_item=sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    data=[]
    for i in similer_item:
        item=[]
        temp_df=books[books['Book-Title']==pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
    return data


#import pickle    
#pickle.dump(pt,open('pt.pkl','wb'))
#pickle.dump(books,open('books.pkl','wb'))
#pickle.dump(similarity_score,open('similarity_score.pkl','wb'))