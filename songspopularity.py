import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from collections import defaultdict
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df_total = pd.read_csv('dat_total.csv')
df_total

def split_data(features, labels):
  xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size = 0.3, random_state = 1)
  return xTrain, xTest, yTrain, yTest

total_xTrain, total_xTest, total_yTrain, total_yTest = split_data(df_total['lyrics'], df_total['popularity'])

def build_vocab_map(df):

    # create a default dict for counting unique vocabs in each email
    vocab_counts = defaultdict(int)

    # for every email
    for i in range(df.shape[0]):
        # create a list of vocabs of the email
        vocabs = df.iloc[i].split(" ")
        # make list into a list of unique vocabs
        vocabs = set(vocabs)
        # for each unique vocabs
        for vocab in vocabs:
            # count unique vocabs in each email
            vocab_counts[vocab] += 1

    # create a dictionary for the vocabulary map
    vocab_map = {}

    # for every vocab and its counts
    for word, count in vocab_counts.items():
        # select the words that appear in at least 30 emails
        if count >= 30:
            vocab_map[word] = count

    if '' in vocab_map:
      vocab_map.pop('', None)

    return vocab_map

total_vocab = build_vocab_map(total_xTrain)
len(total_vocab.keys())

def construct_binary(train_df, vocab_map):
  
    # create a list of words for the vocab map
    frequent_words = list(vocab_map.keys())

    # initialize the binary dataset
    binary_train = np.zeros((train_df.shape[0], len(frequent_words)))

    # for each email
    for i in range(train_df.shape[0]):
        # create a list of unique vocabs in an email
        vocabs = train_df.iloc[i].split(" ")
        vocabs = set(vocabs)

        # for each words in the vocabulary map
        for j in range(len(frequent_words)):
            # if the words in the vocabulary map is in the email
            if frequent_words[j] in vocabs:
                # set vector as 1
                binary_train[i, j] = 1

    return pd.DataFrame(binary_train, columns = frequent_words)

def construct_count(train_df, vocab_map):

    # create a list of words for the vocab map
    frequent_words = list(vocab_map.keys())

    # initialize the count dataset
    count_train = np.zeros((train_df.shape[0], len(frequent_words)))

    # for each email
    for i in range(train_df.shape[0]):
        # create a list of vocabs in an email
        vocabs = train_df.iloc[i].split(" ")

        # for each words in the vocabulary map
        for j in range(len(frequent_words)):
            # count the number of times the jth word appears in the email
            count_train[i,j] = vocabs.count(frequent_words[j])
    
    return pd.DataFrame(count_train, columns = frequent_words)

def construct_all(xTrain, xTest, vocab_list):
  binary_train = construct_binary(xTrain, vocab_list)
  binary_test = construct_binary(xTest, vocab_list)
  count_train = construct_count(xTrain, vocab_list)
  count_test = construct_count(xTest, vocab_list)
  return binary_train, binary_test, count_train, count_test

total_binary_train, total_binary_test, total_count_train, total_count_test = construct_all(total_xTrain, total_xTest, total_vocab)

total_binary_train
total_count_train
total_binary_test
total_count_test

def pos_neg_words(model, word_dict):
  word_list = list(word_dict.keys())
  pos_indices = np.argsort(model.coef_[0])[::-1]
  words_pos = []
  for i in range(15):
      index = pos_indices[i]
      words_pos.append(word_list[index])
  
  neg_indices = np.argsort(model.coef_[0])
  words_neg = []
  for i in range(15):
      index = neg_indices[i]
      words_neg.append(word_list[index])

  return words_pos, words_neg


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

def perceptron_optimal(xTrain, yTrain, word_list):

  max_iters = [1, 10, 20, 40, 80, 120, 150]
  learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]

  plt.figure(figsize=(12, 8))  # Create a figure

  # Plot number of mistakes based on max_iter for different learning rates
  for lr in learning_rates:
    mistakes = []

    for max_iter in max_iters:
      model = Perceptron(random_state=1, max_iter=max_iter, eta0=lr)
      model.fit(xTrain, yTrain)

      predictions = model.predict(xTrain)

      num_mistakes = sum(predictions != yTrain)

      mistakes.append(num_mistakes)

    # Plot the line for current learning rate
    label = 'Learning Rate = {}'.format(lr)
    plt.plot(max_iters, mistakes, '-', label=label)

  plt.xlabel('Max Iterations')
  plt.ylabel('Number of Mistakes')
  plt.legend()
  plt.title('Number of Mistakes vs. Max Iterations for Different Learning Rates')
  plt.show()

perceptron_optimal(total_binary_train, total_yTrain, total_vocab)
perceptron_optimal(total_count_train, total_yTrain, total_vocab)
total_max_iters = [80, 40]
total_learning_rates = [1, 0.01]

def perceptron_accuracy(binary_train, binary_test, count_train, count_test, yTrain, yTest, word_list, max_iters, learning_rates):
  model_binary = Perceptron(random_state = 1, max_iter = max_iters[0], eta0 = learning_rates[0])
  model_binary.fit(binary_train, yTrain)
  model_count = Perceptron(random_state = 1, max_iter = max_iters[1], eta0 = learning_rates[1])
  model_count.fit(count_train, yTrain)
  predictions_binary_train = model_binary.predict(binary_train)
  predictions_binary_test = model_binary.predict(binary_test)
  predictions_count_train = model_count.predict(count_train)
  predictions_count_test = model_count.predict(count_test)
  num_mistakes_binary_train = sum(predictions_binary_train != yTrain)
  num_mistakes_binary_test = sum(predictions_binary_test != yTest)
  num_mistakes_count_train = sum(predictions_count_train != yTrain)
  num_mistakes_count_test = sum(predictions_count_test != yTest)
  print("Number of mistakes on binary train data: ", num_mistakes_binary_train)
  print("Number of mistakes on binary test data: ", num_mistakes_binary_test)
  print("Number of mistakes on count train data: ", num_mistakes_count_train)
  print("Number of mistakes on count test data: ", num_mistakes_count_test)
  binary_train_score = accuracy_score(predictions_binary_train, yTrain)
  print("Accuracy score on binary train data: ", binary_train_score)
  binary_test_score = accuracy_score(predictions_binary_test, yTest)
  print("Accuracy score on binary test data: ", binary_test_score)
  count_train_score = accuracy_score(predictions_count_train, yTrain)
  print("Accuracy score on count train data: ", count_train_score)
  count_test_score = accuracy_score(predictions_count_test, yTest)
  print("Accuracy score on count test data: ", count_test_score)
  
  binary_pos, binary_neg = pos_neg_words(model_binary, word_list)
  count_pos, count_neg = pos_neg_words(model_count, word_list)

  print("15 most positive words for binary model: ")
  print(binary_pos)
  print("15 most negative words for binary model: ")
  print(binary_neg)
  print("15 most positive words for count model: ")
  print(count_pos)
  print("15 most negative words for count model: ")
  print(count_neg)

print("Accuracy score on total dataset")
perceptron_accuracy(total_binary_train, total_binary_test, total_count_train, total_count_test, total_yTrain, total_yTest, total_vocab, total_max_iters, total_learning_rates)

df_edm = pd.read_csv('dat_edm.csv') 
df_latin = pd.read_csv('dat_latin.csv')
df_pop = pd.read_csv('dat_pop.csv')
df_rap = pd.read_csv('dat_rap.csv')
df_rb = pd.read_csv('dat_rb.csv')
df_rock = pd.read_csv('dat_rock.csv')

edm_xTrain, edm_xTest, edm_yTrain, edm_yTest = split_data(df_edm['lyrics'], df_edm['popularity'])
latin_xTrain, latin_xTest, latin_yTrain, latin_yTest = split_data(df_latin['lyrics'], df_latin['popularity'])
pop_xTrain, pop_xTest, pop_yTrain, pop_yTest = split_data(df_pop['lyrics'], df_pop['popularity'])
rap_xTrain, rap_xTest, rap_yTrain, rap_yTest = split_data(df_rap['lyrics'], df_rap['popularity'])
rb_xTrain, rb_xTest, rb_yTrain, rb_yTest = split_data(df_rb['lyrics'], df_rb['popularity'])
rock_xTrain, rock_xTest, rock_yTrain, rock_yTest = split_data(df_rock['lyrics'], df_rock['popularity'])


edm_vocab = build_vocab_map(edm_xTrain)
latin_vocab = build_vocab_map(latin_xTrain)
pop_vocab = build_vocab_map(pop_xTrain)
rap_vocab = build_vocab_map(rap_xTrain)
rb_vocab = build_vocab_map(rb_xTrain)
rock_vocab = build_vocab_map(rock_xTrain)

edm_binary_train, edm_binary_test, edm_count_train, edm_count_test = construct_all(edm_xTrain, edm_xTest, edm_vocab)
latin_binary_train, latin_binary_test, latin_count_train, latin_count_test = construct_all(latin_xTrain, latin_xTest, latin_vocab)
pop_binary_train, pop_binary_test, pop_count_train, pop_count_test = construct_all(pop_xTrain, pop_xTest, pop_vocab)
rap_binary_train, rap_binary_test, rap_count_train, rap_count_test = construct_all(rap_xTrain, rap_xTest, rap_vocab)
rb_binary_train, rb_binary_test, rb_count_train, rb_count_test = construct_all(rb_xTrain, rb_xTest, rb_vocab)
rock_binary_train, rock_binary_test, rock_count_train, rock_count_test = construct_all(rock_xTrain, rock_xTest, rock_vocab)

perceptron_optimal(edm_binary_train, edm_yTrain, edm_vocab)
perceptron_optimal(edm_count_train, edm_yTrain, edm_vocab)
edm_max_iters = [20, 20]
edm_learning_rates = [1, 1]
print("Accuracy score on edm dataset")
perceptron_accuracy(edm_binary_train, edm_binary_test, edm_count_train, edm_count_test, edm_yTrain, edm_yTest, edm_vocab, edm_max_iters, edm_learning_rates)
perceptron_optimal(latin_binary_train, latin_yTrain, latin_vocab)
perceptron_optimal(latin_count_train, latin_yTrain, latin_vocab)

latin_max_iters = [40, 40]
latin_learning_rates = [1, 0.01]

print("Accuracy score on latin dataset")
perceptron_accuracy(latin_binary_train, latin_binary_test, latin_count_train, latin_count_test, latin_yTrain, latin_yTest, latin_vocab, latin_max_iters, latin_learning_rates)
perceptron_optimal(pop_binary_train, pop_yTrain, pop_vocab)
perceptron_optimal(pop_count_train, pop_yTrain, pop_vocab)
pop_max_iters = [40, 40]
pop_learning_rates = [1, 0.01]
print("Accuracy score on pop dataset")
perceptron_accuracy(pop_binary_train, pop_binary_test, pop_count_train, pop_count_test, pop_yTrain, pop_yTest, pop_vocab, pop_max_iters, pop_learning_rates)
perceptron_optimal(rap_binary_train, rap_yTrain, rap_vocab)
perceptron_optimal(rap_count_train, rap_yTrain, rap_vocab)
rap_max_iters = [40, 20]
rap_learning_rates = [1, 1]
print("Accuracy score on rap dataset")
perceptron_accuracy(rap_binary_train, rap_binary_test, rap_count_train, rap_count_test, rap_yTrain, rap_yTest, rap_vocab, rap_max_iters, rap_learning_rates)
perceptron_optimal(rb_binary_train, rb_yTrain, rb_vocab)
perceptron_optimal(rb_count_train, rb_yTrain, rb_vocab)
rb_max_iters = [20, 40]
rb_learning_rates = [0.01, 0.01]
print("Accuracy score on rb dataset")
perceptron_accuracy(rb_binary_train, rb_binary_test, rb_count_train, rb_count_test, rb_yTrain, rb_yTest, rb_vocab, rb_max_iters, rb_learning_rates)
perceptron_optimal(rock_binary_train, rock_yTrain, rock_vocab)
perceptron_optimal(rock_count_train, rock_yTrain, rock_vocab)
rock_max_iters = [40, 20]
rock_learning_rates = [1, 1]
print("Accuracy score on rock dataset")
perceptron_accuracy(rock_binary_train, rock_binary_test, rock_count_train, rock_count_test, rock_yTrain, rock_yTest, rock_vocab, rock_max_iters, rock_learning_rates)


def multinomial(x_train, x_test, y_train, y_test):
    params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0, ],
          'fit_prior': [True, False],
          'class_prior': [None, [0.1,]* 2, ]
         }

    multinomial_nb_grid = GridSearchCV(MultinomialNB(), param_grid=params, n_jobs=-1, cv=5, verbose=5)
    multinomial_nb_grid.fit(x_train, y_train)

    print('Best Accuracy Through Grid Search : {:.3f}'.format(multinomial_nb_grid.best_score_))
    print('Best Parameters : {}\n'.format(multinomial_nb_grid.best_params_))

    # Test the classifier on the test set and print the accuracy
    y_testpred = multinomial_nb_grid.predict(x_test)
    num_testmistakes = sum(y_testpred != y_test)
    y_trainpred = multinomial_nb_grid.predict(x_train)
    num_trainmistakes = sum(y_trainpred != y_train)
    train_accuracy = accuracy_score(y_train, y_trainpred)
    test_accuracy = accuracy_score(y_test, y_testpred)
    print('NB Train Accuracy:', train_accuracy)
    print('NB Number of trainmistakes:', num_trainmistakes)
    print('NB Test Accuracy:', test_accuracy)
    print('NB Number of testmistakes:', num_testmistakes)
    print()

# print('Multinomial EDM Binary')
# multinomial(edm_binary_train, edm_binary_test, edm_yTrain, edm_yTest)
# print('Multinomial EDM Count')
# multinomial(edm_count_train, edm_count_test, edm_yTrain, edm_yTest)
# print('Multinomial Latin Binary')
# multinomial(latin_binary_train, latin_binary_test, latin_yTrain, latin_yTest)
# print('Multinomial Latin Count')
# multinomial(latin_count_train, latin_count_test, latin_yTrain, latin_yTest)
# print('Multinomial Pop Binary')
# multinomial(pop_binary_train, pop_binary_test, pop_yTrain, pop_yTest)
# print('Multinomial Pop Count')
# multinomial(pop_count_train, pop_count_test, pop_yTrain, pop_yTest)
# print('Multinomial Rap Binary')
# multinomial(rap_binary_train, rap_binary_test, rap_yTrain, rap_yTest)
# print('Multinomial Rap Count')
# multinomial(rap_count_train, rap_count_test, rap_yTrain, rap_yTest)
# print('Multinomial RB Binary')
# multinomial(rb_binary_train, rb_binary_test, rb_yTrain, rb_yTest)
# print('Multinomial RB Count')
# multinomial(rb_count_train, rb_count_test, rb_yTrain, rb_yTest)
# print('Multinomial Rock Binary')
# multinomial(rock_binary_train, rock_binary_test, rock_yTrain, rock_yTest)
# print('Multinomial Rock Count')
# multinomial(rock_count_train, rock_count_test, rock_yTrain, rock_yTest)
# print('Multinomial Total Binary')
# multinomial(total_binary_train, total_binary_test, total_yTrain, total_yTest)
# print('Multinomial Total Count')
# multinomial(total_count_train, total_count_test, total_yTrain, total_yTest)

from sklearn import preprocessing

def Logistic(x_train, x_test, y_train, y_test):
  # Logistic Regression
  params = {'C': np.logspace(-3,3,7),
          'penalty': ['l1', 'l2', None, 'elasticnet']}
  logreg_grid = GridSearchCV(LogisticRegression(), param_grid=params, cv=5)
  logreg_grid.fit(x_train, y_train)
  print('Best Accuracy Through Grid Search : {:.3f}'.format(logreg_grid.best_score_))
  print('Best Parameters : {}\n'.format(logreg_grid.best_params_))

  # Test the classifier on the test set and print the accuracy
  y_testpred = logreg_grid.predict(x_test)
  num_testmistakes = sum(y_testpred != y_test)
  y_trainpred = logreg_grid.predict(x_train)
  num_trainmistakes = sum(y_trainpred != y_train)
  train_accuracy = accuracy_score(y_train, y_trainpred)
  test_accuracy = accuracy_score(y_test, y_testpred)
  print('Logistic Regression Train Accuracy:', train_accuracy)
  print('Logistic Regression Number of trainmistakes:', num_trainmistakes)
  print('Logistic Regression Test Accuracy:', test_accuracy)
  print('Logistic Regression Number of testmistakes:', num_testmistakes)
  print()


# print('Logistic EDM Binary')
# Logistic(edm_binary_train, edm_binary_test, edm_yTrain, edm_yTest)
# print('Logistic EDM Count')
# Logistic(edm_count_train, edm_count_test, edm_yTrain, edm_yTest)
# print('Logistic Latin Binary')
# Logistic(latin_binary_train, latin_binary_test, latin_yTrain, latin_yTest)
# print('Logistic Latin Count')
# Logistic(latin_count_train, latin_count_test, latin_yTrain, latin_yTest)
# print('Logistic Pop Binary')
# Logistic(pop_binary_train, pop_binary_test, pop_yTrain, pop_yTest)
# print('Logistic Pop Count')
# Logistic(pop_count_train, pop_count_test, pop_yTrain, pop_yTest)
# print('Logistic Rap Binary')
# Logistic(rap_binary_train, rap_binary_test, rap_yTrain, rap_yTest)
# print('Logistic Rap Count')
# Logistic(rap_count_train, rap_count_test, rap_yTrain, rap_yTest)
# print('Logistic RB Binary')
# Logistic(rb_binary_train, rb_binary_test, rb_yTrain, rb_yTest)
# print('Logistic RB Count')
# Logistic(rb_count_train, rb_count_test, rb_yTrain, rb_yTest)
# print('Logistic Rock Binary')
# Logistic(rock_binary_train, rock_binary_test, rock_yTrain, rock_yTest)
# print('Logistic Rock Count')
# Logistic(rock_count_train, rock_count_test, rock_yTrain, rock_yTest)
# print('Logistic Total Binary')
# Logistic(total_binary_train, total_binary_test, total_yTrain, total_yTest)
# print('Logistic Total Count')
# Logistic(total_count_train, total_count_test, total_yTrain, total_yTest)

import statsmodels.api as sm

def graph(y_train, X_train):
  logit_model = sm.Logit(y_train, X_train)
  result = logit_model.fit_regularized(method='l1',alpha=5)
  print(result.summary())

print('Multinomial EDM Binary')
graph(edm_binary_train, edm_yTrain)

print('Multinomial Latin Binary')
graph(latin_binary_train, latin_yTrain)

print('Multinomial Pop Binary')
graph(pop_binary_train, pop_yTrain)

print('Multinomial Rap Binary')
graph(rap_binary_train, rap_yTrain)

print('Multinomial RB Binary')
graph(rb_binary_train, rb_yTrain)

print('Multinomial Rock Binary')
graph(rock_binary_train, rock_yTrain)

print('Multinomial Total Binary')
graph(total_binary_train, total_yTrain)


print('Logistic EDM Binary')
graph(edm_binary_train, edm_yTrain)

print('Logistic Latin Binary')
graph(latin_binary_train, latin_binary_test, latin_yTrain, latin_yTest)

print('Logistic Pop Binary')
graph(pop_binary_train, pop_yTrain)

print('Logistic Rap Binary')
graph(rap_binary_train, rap_yTrain)

print('Logistic RB Binary')
graph(rb_binary_train, rb_yTrain)

print('Logistic Rock Binary')
graph(rock_binary_train, rock_yTrain)

print('Logistic Total Binary')
graph(total_binary_train, total_yTrain)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('spotify_songs.csv')
df.head()

X = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]
y = df['track_popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error: {:.2f}".format(mse))

print("R^2 Score: {:.2f}".format(r2))

sns.scatterplot(x='danceability', y='track_popularity', data=df)
plt.show()

sns.scatterplot(x='energy', y='track_popularity', data=df)
plt.show()

sns.scatterplot(x='key', y='track_popularity', data=df)
plt.show()

sns.scatterplot(x='loudness', y='track_popularity', data=df)
plt.show()

sns.scatterplot(x='mode', y='track_popularity', data=df)
plt.show()

sns.scatterplot(x='speechiness', y='track_popularity', data=df)
plt.show()

sns.scatterplot(x='acousticness', y='track_popularity', data=df)
plt.show()

sns.scatterplot(x='instrumentalness', y='track_popularity', data=df)
plt.show()

sns.scatterplot(x='liveness', y='track_popularity', data=df)
plt.show()

sns.scatterplot(x='valence', y='track_popularity', data=df)
plt.show()

sns.scatterplot(x='tempo', y='track_popularity', data=df)
plt.show()

sns.scatterplot(x='duration_ms', y='track_popularity', data=df)
plt.show()

corr_matrix = df.corr()
corr_matrix

corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, annot_kws={"fontsize":12}, ax=ax)
plt.show()

residuals = y_test - y_pred
residuals

fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(residuals, ax=ax)
plt.show()

y_pred

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dflyrics = pd.read_csv('dat.csv')
dflyrics.head()

target = 'track_popularity'
selected_features = []
df_filtered = dflyrics.loc[:, ~dflyrics.columns.isin(['track_id', 'track_name', 'track_artist', 'playlist_genre'])]
df_filtered

import plotly.express as px
from sklearn.decomposition import PCA


pca = PCA(n_components=3)
components = pca.fit_transform(df_filtered)

total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=dflyrics['track_popularity'],
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3',  'color': 'Popularity'}
)
fig.show()

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# assume df is the DataFrame with 768-dimensional input features and 'track_popularity' as the output variable
X = df_filtered
y = dflyrics['track_popularity']

# apply PCA to reduce the number of dimensions to 3
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)

# fit a linear regression model on the training set
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on the test set
y_pred = model.predict(X_test)

# visualize the predictions against the actual values in a scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Track Popularity")
plt.ylabel("Predicted Track Popularity")
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# apply PCA to reduce the number of dimensions to 3
X = df_filtered
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
y = dflyrics['track_popularity']

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)

# fit a linear regression model on the training set
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on the training set
y_pred = model.predict(X_train)

# create a 3D scatter plot of the training set input features
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='coolwarm')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('Actual Track Popularity')
plt.show()

# create a 3D scatter plot of the training set input features with the predicted values as a regression plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='coolwarm')
x1, x2 = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 10), 
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 10))
y_pred_plane = model.predict(np.c_[x1.ravel(), x2.ravel(), np.zeros_like(x1.ravel())])
y_pred_plane = y_pred_plane.reshape(x1.shape)
ax.plot_surface(x1, x2, y_pred_plane, alpha=0.5, cmap='coolwarm')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('Track Popularity')
ax.set_title('Predicted Track Popularity')
plt.show()


X_pca
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: {:.2f}".format(mse))
print("R^2 Score: {:.2f}".format(r2))


residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

plt.hist(residuals, bins=20)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

sns.boxplot(y=residuals)
plt.title('Box Plot of Residuals')
plt.show()