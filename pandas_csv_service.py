import pandas as pd
import numpy as np # Linear algebra and pandas compatibility
import pandas as pd # Data management, and dataframes
from sklearn.model_selection import train_test_split # Splits dataset into training|testing sets
from sklearn.tree import DecisionTreeRegressor # Decision Tree Model
from sklearn.ensemble import RandomForestRegressor # Random Forest Model
from sklearn.metrics import mean_absolute_error # MAE, measuring loss
import matplotlib.pyplot as plt # Graphing library to visualize the data/correlations
import seaborn as sns #Heatmap
from sklearn.linear_model import LogisticRegression

tracks = pd.read_csv('tracks.csv')
artists = pd.read_csv('artists.csv')

def find_most_popular_artist(artist_ids):
    max_popularity = 0
    if len(artist_ids) > 26:    #svaki id u sirovom formatu ima 26 karaktera
        singers = artist_ids[1:len(artist_ids)-1].split(', ')   #ako ima vise od jednog izvodjaca, uzmi svakog od njih
        counter = 0
        for singer in singers:      #proveri svima popularnost
            if counter == 0:
                found_singer = artists.loc[artists['id'] == singer[0:22]]   #3BiJGZsyX9sJchTqcSA7Su'    prvi id bude u ovom obliku
            elif counter == len(singers) - 1:
                found_singer = artists.loc[artists['id'] == singer[1:]]     #'3BiJGZsyX9sJchTqcSA7Su   poslednji id u ovom obliku
            else:
                found_singer = artists.loc[artists['id'] == singer[1:23]]   #'3BiJGZsyX9sJchTqcSA7Su    svi izmedju u ovom obliku
            counter += 1
            popularity_value = found_singer['popularity'].item()
            if popularity_value > max_popularity:
                max_popularity = popularity_value       #selektovanje najpopularnijeg od vise ivodjaca
    else:                       #pesma ima jednog izvodjaca
        found_singer = artists.loc[artists['id'] == artist_ids[1:23]]
        max_popularity = found_singer['popularity'].item()
    return max_popularity

def find_intersection():
    artists_popularity = []     #lista koja ce predstavljati novu kolonu dataseta
    merged = tracks.head(100)   #uzeto 100 prvih samo da se proveri da li radi
    counter = 0
    for song in tracks.values:  #df.insert(0, "col1", [100, 100], allow_duplicates=True)
        if counter >= 100:
            break
        artists_popularity.append(find_most_popular_artist(song[6][1:len(song[6]) - 1]))
        counter += 1
        print(counter)
    merged.release_date = merged.release_date.str[0:4]      #pretvaranje datuma objave pesme u godinu, zbog lakse obrade podataka
    merged.insert(20, "artist_popularity", artists_popularity, allow_duplicates=True)
    merged.to_csv('merged.csv')     #export dataseta pesama zajedno sa popularnosti izvodjaca u .csv fajl
    print(artists_popularity)
    print(merged.values)
