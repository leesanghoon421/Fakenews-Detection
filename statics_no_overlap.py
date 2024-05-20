import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm

# Load data
df = pd.read_csv('fake_true.csv')
df.dropna(subset = ['text', 'title'], inplace = True)
df['text'] = df['title'] + ' ' + df['text']

# Define stopwords
stop_words = set(stopwords.words('english'))

def get_top_n_words(corpus, n=100):
    # Initialize a Counter object
    freq_dist = Counter()
    
    for text in tqdm(corpus, desc="Processing text"):
        # Convert text to lowercase
        text = text.lower()

        # Tokenize and filter out stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stop_words]

        # Update the frequency distribution
        freq_dist.update(tokens)
        
    return freq_dist.most_common(n)
    

# Separate fake and true news
fake_news = df[df['label'] == 0]['text']
true_news = df[df['label'] == 1]['text']

# Get top 100 words
top_fake = dict(get_top_n_words(fake_news, 100))
top_true = dict(get_top_n_words(true_news, 100))

# Remove overlapping words
unique_fake = {word: freq for word, freq in top_fake.items() if word not in top_true}
unique_true = {word: freq for word, freq in top_true.items() if word not in top_fake}

# Match the length of lists
len_min = min(len(unique_fake), len(unique_true))
unique_fake = dict(sorted(unique_fake.items(), key=lambda item: item[1], reverse=True)[:len_min])
unique_true = dict(sorted(unique_true.items(), key=lambda item: item[1], reverse=True)[:len_min])

print("Top words in fake news (no overlap):")
print(list(unique_fake.keys()))
print("\nTop words in true news (no overlap):")
print(list(unique_true.keys()))

def create_wordcloud(words, title):
    print(f"Creating wordcloud: {title}")
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=300).generate_from_frequencies(words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    plt.show()

# Create word clouds
create_wordcloud(unique_fake, "Fake News")
create_wordcloud(unique_true, "True News")
