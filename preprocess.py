import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the DataFrame
df = pd.read_csv('spotify_songs.csv')

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/LaBSE')

# Define batch size
batch_size = 128

# Create a list to store the embeddings
lyrics_embeddings = []

# Encode the lyrics in batches
for i in range(0, len(df), batch_size):
    batch = df['lyrics'][i:i+batch_size].tolist()
    batch_embeddings = model.encode(batch)
    lyrics_embeddings.extend(batch_embeddings)
    print('finished batch{}'.format(i))

# Create a DataFrame with a column for each embedding dimension
lyrics_dim = pd.DataFrame(lyrics_embeddings, columns=[f'lyrics_dim_{i}' for i in range(768)])

# Concatenate the new DataFrame with the original input DataFrame
new_df = pd.concat([df[['track_popularity', 'track_id', 'track_name', 'track_artist', 'playlist_genre']], lyrics_dim], axis=1)

# Save the new DataFrame to a CSV file
new_df.to_csv('dat.csv', index=False)