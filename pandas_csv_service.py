import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Random Forest Model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from yellowbrick.model_selection import FeatureImportances # Correlation finding

tracks = pd.read_csv('tracks.csv')
artists = pd.read_csv('artists.csv')

def find_most_popular_artist(artist_ids):
    max_popularity = 0
    if len(artist_ids) > 26:    #svaki id u sirovom formatu ima 26 karaktera
        #print(len(artist_ids))
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
            if len(found_singer) == 1:
                popularity_value = found_singer['popularity'].item()
                if popularity_value > max_popularity:
                    max_popularity = popularity_value       #selektovanje najpopularnijeg od vise ivodjaca
    else:                       #pesma ima jednog izvodjaca
        #print(len(artist_ids))
        found_singer = artists.loc[artists['id'] == artist_ids[1:23]]
        if len(found_singer) == 1:
            max_popularity = found_singer['popularity'].item()
    return max_popularity

def merge_tracks_and_artists():
    artists_popularity = []     #lista koja ce predstavljati novu kolonu dataseta
    shuffled_tracks = tracks.sample(frac=1)
    merged = shuffled_tracks.head(20000)
    counter = 0
    for song in shuffled_tracks.values:
        if counter >= 20000:
            break
        artists_popularity.append(find_most_popular_artist(song[6][1:len(song[6]) - 1]))
        counter += 1
        print(counter)
    merged.release_date = merged.release_date.str[0:4]      #pretvaranje datuma objave pesme u godinu, zbog lakse obrade podataka
    merged.insert(20, "artist_popularity", artists_popularity, allow_duplicates=True)
    merged.to_csv('merged-20000.csv')     #export dataseta pesama zajedno sa popularnosti izvodjaca u .csv fajl
    #print(artists_popularity)
    #print(merged.values)

def show_importances():
    df = pd.read_csv('merged-20000.csv')
    y = df.popularity
    features = ['duration_ms', 'explicit', 'release_date',
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'time_signature', 'artist_popularity']
    X = df[features]
    print(X.head())

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

    rfr_model = RandomForestRegressor()
    feature_importances = FeatureImportances(rfr_model)
    feature_importances.fit(train_X, train_y)
    feature_importances.show()

def logistic_regression_prediction():
    df = pd.read_csv('merged-5000.csv')
    y = df.popularity
    features = ['duration_ms', 'explicit', 'release_date',
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'time_signature', 'artist_popularity']
    X = df[features]
    print(X.head())

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
    lr_model = LogisticRegression(random_state=0)
    lr_model.fit(train_X, train_y)
    val_preds1 = lr_model.predict(test_X)
    error = mean_squared_error(test_y, val_preds1)
    print(f'Mean squared error of this model: {math.sqrt(error):.3f}')

def random_forest_prediction():
    df = pd.read_csv('merged-20000.csv')
    print(df.info())
    y = df.popularity
    features = ['duration_ms', 'explicit', 'release_date',
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'time_signature', 'artist_popularity']
    X = df[features]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
    rf_model = RandomForestRegressor(random_state=0)
    rf_model.fit(train_X, train_y)
    val_preds1 = rf_model.predict(test_X)
    mse_error = mean_squared_error(test_y, val_preds1)
    mae_error = mean_absolute_error(test_y, val_preds1)
    r2 = r2_score(test_y, val_preds1)
    print(f' Root mean squared error of this model: {math.sqrt(mse_error):.3f}')
    print(f' Mean absolute error of this model: {mae_error:.3f}')
    print(f' R2 score of this model: {r2*100:.3f} %')


def show_info():
    merged = pd.read_csv('merged-20000.csv')
    merged.describe().transpose().to_csv('statistics-merged-20000.csv')

def format_dataset():
    scaler = MinMaxScaler()
