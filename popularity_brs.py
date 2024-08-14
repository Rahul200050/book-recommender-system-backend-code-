# I started making popularity base books recommender system so because ye recommand karega books but isme hum machine ko instruction learn krwatey h,
# prediction nahi prediction ma hum machine ko training data learn krwate h and uske basis pr prediction krwate h but isme (DATA--->MLM(INSTRUCTION)--->OUTPUT)

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
# (2) har book ko kitni number of rating mili isko extract lr rahe h RWBN file se by (groupby) and book-rating ka name change krke num_rating rakh rahe h and is file ka name num_rating_df h
num_rating_df = rating_with_books_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
# (3) book rating ke column ki rows ma jaha jaha value NaN h usko hata raha h taaki average of rating according to book nikaalte huy error na aay, waise to ye feature engineering ma hona tha but ye bhool gay isliye yaha kr diya
# Ensure 'Book-Rating' is numeric, non-numeric values will be converted to NaN, Drop rows with NaN values in 'Book-Rating' and uske baad average rating 
rating_with_books_name['Book-Rating'] = pd.to_numeric(rating_with_books_name['Book-Rating'], errors='coerce')
rating_with_books_name.dropna(subset=['Book-Rating'], inplace=True)
avg_rating_df = rating_with_books_name.groupby('Book-Title', as_index=False)['Book-Rating'].mean()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
# (4) ab humne jo uper books ke according average rating and number of rating wali file banaai h usko merge karegay but book title waley column ko ek hi baar show karegay and DA krne ke baad finally yue last dataset hoga jo direct use hoga machine learning modal ma. 
popular_df=num_rating_df.merge(avg_rating_df,on='Book-Title')

#recommander system bhot types ke hote h ek type h POPULARITY BASE RECOMMMANDER SYSTEM ,popularity base recommander system :- ye trending chize recommand karata h jaise example- youtube(trending)
# MACHINE LEARNING MODAL:-
# (isme hum MACHINE ko ye learn kara rahe h ki dataset ma jin books ki rating 250 ya usse jada h unme se jin 50 books ka average_rating sabse jada ho un books ko output krdo) 

popular_df=popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(50)
popular_df=popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]

print(popular_df['Image-URL-M'][0])
#import pickle
#pickle.dump(popular_df,open('popular.pkl','wb'))
