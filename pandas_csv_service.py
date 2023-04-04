import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Random Forest Model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from yellowbrick.model_selection import FeatureImportances # Correlation finding
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

tracks = pd.read_csv('tracks.csv')
artists = pd.read_csv('artists.csv')
final_tracks = pd.read_csv('final-shuffled-20000.csv')

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
    id = []
    name = []
    popularity = []
    duration_ms = []
    explicit = []
    artists = []
    id_artists = []
    release_date = []
    danceability = []
    energy = []
    key = []
    loudness = []
    mode = []
    speechiness = []
    acousticness = []
    instrumentalness = []
    liveness = []
    valence = []
    tempo = []
    time_signature = []
    artist_popularity = []     #lista koja ce predstavljati novu kolonu dataseta
    popularity_class = []       #pomocna lista koja ce sluziti za oversampling: klasa 1 - popularnost(0-20), klasa 2 - popularnost(21-40)...
    shuffled_tracks = final_tracks.sample(frac=1)
    zero_to_ten = 0
    eleven_to_twenty = 0
    twenty_one_to_thirty = 0
    thirty_one_to_forty = 0
    forty_one_to_fifty = 0
    fifty_one_to_sixty = 0
    sixty_one_to_seventy = 0
    seventy_one_to_eighty = 0
    merged = pd.DataFrame()
    counter = 0
    popular_counter = 0
    popular_tracks = tracks.sort_values('popularity', ascending=False)
    for song in popular_tracks.values:        #sve pesme sa popularnoscu preko 80
        if popular_counter > 736:
            break
        id.append(song[0])
        name.append(song[1])
        popularity.append(song[2])
        duration_ms.append(song[3])
        explicit.append(song[4])
        artists.append(song[5])
        id_artists.append(song[6])
        release_date.append(song[7])
        danceability.append(song[8])
        energy.append(song[9])
        key.append(song[10])
        loudness.append(song[11])
        mode.append(song[12])
        speechiness.append(song[13])
        acousticness.append(song[14])
        instrumentalness.append(song[15])
        liveness.append(song[16])
        valence.append(song[17])
        tempo.append(song[18])
        time_signature.append(song[19])
        artist_popularity.append(find_most_popular_artist(song[6][1:len(song[6]) - 1]))  # nalazenje popularnosti umetnika
        popularity_class.append(5)
        popular_counter += 1
        print(popular_counter)
    previous_counter = counter
    for song in shuffled_tracks.values:
        if counter >= 2944:
            break
        song_popularity = song[3]
        if 0 <= song_popularity < 11 and zero_to_ten < 368:
            zero_to_ten += 1
            popularity_class.append(1)
            counter += 1
        elif 11 <= song_popularity < 21 and eleven_to_twenty < 368:
            eleven_to_twenty += 1
            popularity_class.append(1)
            counter += 1
        elif 21 <= song_popularity < 31 and twenty_one_to_thirty < 368:
            twenty_one_to_thirty += 1
            popularity_class.append(2)
            counter += 1
        elif 31 <= song_popularity < 41 and thirty_one_to_forty < 368:
            thirty_one_to_forty += 1
            popularity_class.append(2)
            counter += 1
        elif 41 <= song_popularity < 51 and forty_one_to_fifty < 368:
            forty_one_to_fifty += 1
            popularity_class.append(3)
            counter += 1
        elif 51 <= song_popularity < 61 and fifty_one_to_sixty < 368:
            fifty_one_to_sixty += 1
            popularity_class.append(3)
            counter += 1
        elif 61 <= song_popularity < 71 and sixty_one_to_seventy < 368:
            sixty_one_to_seventy += 1
            popularity_class.append(4)
            counter += 1
        elif 71 <= song_popularity < 81 and seventy_one_to_eighty < 368:
            seventy_one_to_eighty += 1
            popularity_class.append(4)
            counter += 1
        if previous_counter != counter:
            id.append(song[1])
            name.append(song[2])
            popularity.append(song[3])
            duration_ms.append(song[4])
            explicit.append(song[5])
            artists.append(song[6])
            id_artists.append(song[7])
            release_date.append(str(song[8]))
            danceability.append(song[9])
            energy.append(song[10])
            key.append(song[11])
            loudness.append(song[12])
            mode.append(song[13])
            speechiness.append(song[14])
            acousticness.append(song[15])
            instrumentalness.append(song[16])
            liveness.append(song[17])
            valence.append(song[18])
            tempo.append(song[19])
            time_signature.append(song[20])
            artist_popularity.append(song[21])        #nalazenje popularnosti umetnika
            print(counter)
            previous_counter = counter
    print(release_date)
    merged.insert(0, "id", id, allow_duplicates=True)
    merged.insert(1, "name", name, allow_duplicates=True)
    merged.insert(2, "popularity", popularity, allow_duplicates=True)
    merged.insert(3, "duration_ms", duration_ms, allow_duplicates=True)
    merged.insert(4, "explicit", explicit, allow_duplicates=True)
    merged.insert(5, "artists", artists, allow_duplicates=True)
    merged.insert(6, "id_artists", id_artists, allow_duplicates=True)
    merged.insert(7, "release_date", release_date, allow_duplicates=True)
    merged.insert(8, "danceability", danceability, allow_duplicates=True)
    merged.insert(9, "energy", energy, allow_duplicates=True)
    merged.insert(10, "key", key, allow_duplicates=True)
    merged.insert(11, "loudness", loudness, allow_duplicates=True)
    merged.insert(12, "mode", mode, allow_duplicates=True)
    merged.insert(13, "speechiness", speechiness, allow_duplicates=True)
    merged.insert(14, "acousticness", acousticness, allow_duplicates=True)
    merged.insert(15, "instrumentalness", instrumentalness, allow_duplicates=True)
    merged.insert(16, "liveness", liveness, allow_duplicates=True)
    merged.insert(17, "valence", valence, allow_duplicates=True)
    merged.insert(18, "tempo", tempo, allow_duplicates=True)
    merged.insert(19, "time_signature", time_signature, allow_duplicates=True)
    merged.insert(20, "artist_popularity", artist_popularity, allow_duplicates=True)
    merged.insert(21, "popularity_class", popularity_class, allow_duplicates=True)
    print(merged.release_date)
    merged.release_date = merged.release_date.str[0:4]      #pretvaranje datuma objave pesme u godinu, zbog lakse obrade podataka
    print(merged.release_date)
    merged_shuffled = merged.sample(frac=1)     #shuffle
    merged_shuffled.to_csv('undersampled-even-shuffled-20000.csv')     #export dataseta pesama zajedno sa popularnosti izvodjaca u .csv fajl
    #print(artists_popularity)
    #print(merged.values)

def show_importances():
    df = pd.read_csv('final-shuffled-20000.csv')
    y = df.popularity
    features = ['duration_ms', 'explicit', 'release_date',
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'time_signature', 'artist_popularity']
    X = df[features]
    print(X.head())

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
    # model.fit(X, y)
    # importance = model.coef_
    # summarize feature importance
    # for i, v in enumerate(importance):
    #    print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.show()


def linear_regression_prediction():
    df = pd.read_csv('final-shuffled-20000.csv')
    y = df.popularity
    features = ['duration_ms', 'explicit', 'release_date',
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'time_signature', 'artist_popularity']
    X = df[features]
    X['duration_ms'] = (X['duration_ms'] - X['duration_ms'].min()) / (X['duration_ms'].max() - X['duration_ms'].min())
    X['release_date'] = (X['release_date'] - X['release_date'].min()) / (X['release_date'].max() - X['release_date'].min())
    X['key'] = (X['key'] - X['key'].min()) / (X['key'].max() - X['key'].min())
    X['tempo'] = (X['tempo'] - X['tempo'].min()) / (X['tempo'].max() - X['tempo'].min())
    X['time_signature'] = (X['time_signature'] - X['time_signature'].min()) / (X['time_signature'].max() - X['time_signature'].min())
    X['speechiness'] = (X['speechiness'] - X['speechiness'].min()) / (X['speechiness'].max() - X['speechiness'].min())
    X['loudness'] = X['loudness'] + X['loudness'].min()
    X['loudness'] = (X['loudness'] - X['loudness'].min()) / (X['loudness'].max() - X['loudness'].min())
    X['artist_popularity'] = (X['artist_popularity'] - X['artist_popularity'].min())/(X['artist_popularity'].max() - X['artist_popularity'].min())
    print(X)
    print(X.info())
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    lr_model = LinearRegression()
    lr_model.fit(train_X, train_y)
    val_preds1 = lr_model.predict(test_X)
    mse_error = mean_squared_error(test_y, val_preds1)
    mae_error = mean_absolute_error(test_y, val_preds1)
    r2 = r2_score(test_y, val_preds1)
    print(f' Root mean squared error of this model: {math.sqrt(mse_error):.3f}')
    print(f' Mean absolute error of this model: {mae_error:.3f}')
    print(f' R2 score of this model: {r2 * 100:.3f} %')
    feature_importances = FeatureImportances(lr_model)
    feature_importances.fit(train_X, train_y)
    feature_importances.show()

def random_forest_prediction():
    df = pd.read_csv('undersampled-even-shuffled-20000.csv')
    print(df.info())
    y = df.popularity
    features = ['duration_ms', 'explicit', 'release_date',
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'time_signature', 'artist_popularity']
    X = df[features]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    rf_model = RandomForestRegressor(random_state=0)
    rf_model.fit(train_X, train_y)
    val_preds1 = rf_model.predict(test_X)
    mse_error = mean_squared_error(test_y, val_preds1)
    mae_error = mean_absolute_error(test_y, val_preds1)
    r2 = r2_score(test_y, val_preds1)
    print(f' Root mean squared error of this model: {math.sqrt(mse_error):.3f}')
    print(f' Mean absolute error of this model: {mae_error:.3f}')
    print(f' R2 score of this model: {r2*100:.3f} %')
    feature_importances = FeatureImportances(rf_model)
    feature_importances.fit(train_X, train_y)
    feature_importances.show()


def show_info():
    #merged = pd.read_csv('tracks.csv')
    #merged.describe().transpose().to_csv('statistics-tracks.csv')
    tracks.release_date = tracks.release_date.str[0:4]
    tracks.release_date = tracks.release_date.astype(int)
    sns.heatmap(tracks.corr(), cmap='icefire');

    plt.show()

def show_histograms():
    merged = pd.read_csv('undersampled-even-shuffled-20000.csv')
    merged.hist(bins=20, color='orange', figsize=(20, 14))
    plt.show()

def count_very_popular():
    df = pd.read_csv('tracks.csv')
    features = ['release_date', 'popularity',
                'danceability', 'energy']
    X = df[features]
    X = X[(X.popularity > 80)]          #736 pesama sa popularnoscu iznad 80
    print(X)

def test():
    popular_tracks = tracks.sort_values('popularity', ascending=False)
    print(popular_tracks)

def neural_network():
    df = pd.read_csv('undersampled-even-shuffled-20000.csv')

    y = df.popularity
    features = ['duration_ms', 'explicit', 'release_date',
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'time_signature', 'artist_popularity']
    X = df[features]
    X['duration_ms'] = (X['duration_ms'] - X['duration_ms'].min()) / (X['duration_ms'].max() - X['duration_ms'].min())
    X['release_date'] = (X['release_date'] - X['release_date'].min()) / (
                X['release_date'].max() - X['release_date'].min())
    X['key'] = (X['key'] - X['key'].min()) / (X['key'].max() - X['key'].min())
    X['tempo'] = (X['tempo'] - X['tempo'].min()) / (X['tempo'].max() - X['tempo'].min())
    X['time_signature'] = (X['time_signature'] - X['time_signature'].min()) / (
                X['time_signature'].max() - X['time_signature'].min())
    X['speechiness'] = (X['speechiness'] - X['speechiness'].min()) / (X['speechiness'].max() - X['speechiness'].min())
    X['loudness'] = X['loudness'] + X['loudness'].min()
    X['loudness'] = (X['loudness'] - X['loudness'].min()) / (X['loudness'].max() - X['loudness'].min())
    X['artist_popularity'] = (X['artist_popularity'] - X['artist_popularity'].min()) / (
                X['artist_popularity'].max() - X['artist_popularity'].min())

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    label_encoder_y = LabelEncoder()
    train_y = label_encoder_y.fit_transform(train_y)
    test_y = label_encoder_y.fit_transform(test_y)
    #onehot_encoder = OneHotEncoder(sparse=False)
    #train_y = onehot_encoder.fit_transform(train_y.reshape(len(train_y), 1))
    print(train_y.shape)
    model = Sequential()
    model.add(Dense(train_X.shape[1], input_dim=train_X.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    model.fit(train_X, train_y, epochs=1000, batch_size=32, validation_data=(test_X, test_y))

    ynew = model.predict(test_X)

    score = model.evaluate(test_X, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    for i in range(len(test_X)):
        print(test_y[i], ynew[i])