# import packages
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pickle
import fastparquet
import dask.dataframe as dd
#from surprise import Reader, Dataset, KNNBasic, accuracy
#from surprise.model_selection import train_test_split


# import data
links = pd.read_csv("./data1/links.csv")
movies = pd.read_csv("./data1/movies.csv")
ratings = pd.read_csv("./data1/ratings.csv")
tags = pd.read_csv("./data1/tags.csv")

# predictions: Read Parquet file using Dask
p_dd = dd.read_parquet("./data1/predictions.parquet")

# import model
#loaded_model = pickle.load(open("./data1/knn_surprise_recommender.sav", 'rb'))




# SETTINGS:
# number of recommendations n
n=5
# number of shown recommenation columns 
num_columns = 5






# POPULARITY FUNCTION

# getting 10 most popular titles
def pub_rec(movies_df, ratings_df, links_df):
    df = pd.merge(movies_df,
        ratings_df, 
        how="inner",
        on="movieId").groupby("movieId").agg({"rating":["mean","count"]}).sort_values(("rating","mean"), ascending=False)
    df.columns = ["rating_mean", "rating_count"]

    df = pd.merge(df[df["rating_count"]>150],
              movies_df,
              how="left",
              on="movieId")
    df_out = pd.merge(df, 
                      links_df,
                      how="left",
                      on="movieId")
    return(df_out[0:5])
pub_recommendations = pub_rec(movies, ratings, links)


# retrieving imdb ids based on 
pub_rec_df = []
imdb_df = []
for i in range(5):  
    text_variable = pub_recommendations.loc[i, "title"]  
    imdb = pub_recommendations.loc[i, "imdbId"]
    pub_rec_df.append(text_variable)
    imdb_df.append(imdb)


# pub recommendations imdb pics
def extract_image_url(imdb_id):
    # First, try with the IMDb URL having "/tt00" in the link
    html_url = f"https://www.imdb.com/title/tt00{imdb_id}/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    
    # Send an HTTP GET request to fetch the HTML content
    response = requests.get(html_url, headers=headers)

    # Check the response status code
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the <meta> tag with the property "og:image"
        meta_tag = soup.find("meta", {"property": "og:image"})

        if meta_tag:
            pic_url = meta_tag["content"]
            return pic_url
        else:
            return "Image URL not found on the page"
            
    elif response.status_code == 404:
        # If the page is not found (404), try with the IMDb URL having "/tt0" in the link
        html_url = f"https://www.imdb.com/title/tt0{imdb_id}/"
        response = requests.get(html_url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            meta_tag = soup.find("meta", {"property": "og:image"})

            if meta_tag:
                pic_url = meta_tag["content"]
                return pic_url
            else:
                return "Image URL not found on the page"

    elif response.status_code == 404:
        # If the page is not found (404), try with the IMDb URL having "/tt0" in the link
        html_url = f"https://www.imdb.com/title/tt{imdb_id}/"
        response = requests.get(html_url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            meta_tag = soup.find("meta", {"property": "og:image"})

            if meta_tag:
                pic_url = meta_tag["content"]
                return pic_url
            else:
                return "Image URL not found on the page"


        else:
            return f"Fai1. Status code: {response.status_code}"
    else:
        return f"Fail2. Status code"









# SURPRISE DATA

def get_top_n_for_user(p_dd, user_id, n=5):
    iids = []
    p = p_dd.loc[p_dd["uid"]==user_id,:].compute().sort_values(by="est", ascending=False)[0:n]
    iids = p["iid"].values.tolist()
    top_n_df = pd.merge(links.loc[links["movieId"].isin(iids),:], 
        movies,
        how="left",
        on="movieId")

    return top_n_df











### THE PAGE ###

st.title("Your Movie Recommendations")

st.write("""
Here you can see your presonalized movie recommendations based on your previous movies. Just log in with your User ID and enjoy! 
""")
userId = st.text_input("Log In with User ID:")
#userId = 123


st.markdown("---")



## PUBLICITY RECOMMENDER
st.write("### Trending")



# Iterate through the data and display information
for i in range(num_columns):
    if i % num_columns == 0:
        col0, col1, col2, col3, col4 = st.columns(5)
    
    #pic_path = f"C:/Users/daedlow/Documents/jupyter_notebook/recommender_systems/pic_db/{imdb_df[i]}_pic.csv"

    pic_url = extract_image_url(imdb_df[i])
    
    # If extract_image_url returns None, try with the IMDb URL having "/tt0"
    if pic_url is None:
        pic_url = extract_image_url(imdb_df[i])

    with locals()[f"col{i % num_columns}"]:
        st.write(pub_rec_df[i])
        st.image(pic_url)
        #st.image(pic_path)




### USER-BASED RECOMMENDER
st.write("### Based on what other people liked")

userId = st.text_input("Log In with User ID:")

if userId:
    top_n = get_top_n_for_user(p_dd, int(userId), n)
    for i in range(num_columns):
        imdb=top_n.loc[i, "imdbId"]
        if i % num_columns == 0:
            col0, col1, col2, col3, col4 = st.columns(5)
    
        #pic_path = f"C:/Users/daedlow/Documents/jupyter_notebook/recommender_systems/pic_db/{imdb}_pic.csv"

        pic_url = extract_image_url(imdb)
         

        with locals()[f"col{i % num_columns}"]:
            st.write(top_n.loc[i, "title"])
            if pic_url:
                st.image(pic_url, column_width=True)
            if pic_url is None:
                st.write("no picture available")





### ITEM-BASED RECOMMENDER
st.write("### Based on what you saw before")




