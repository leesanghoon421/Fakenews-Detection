import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm
import seaborn as sns

# Load data
df = pd.read_csv('fake_true.csv')
df.dropna(subset = ['text', 'title'], inplace = True)
df['text'] = df['title'] + ' ' + df['text']

# Add num_words column to the dataframe
df['num_words'] = df['text'].apply(lambda x: len(word_tokenize(x)))

plt.figure(figsize = (10,6))
sns.countplot(x = df['label'], alpha = 0.8)
plt.title('Distribution of Fake - 0 /Real - 1 News')
plt.show()

plt.figure(figsize = (20,6))
sns.histplot(df['num_words'], bins = range(1, 3000, 50), alpha = 0.8)
plt.title('Distribution of the News Words count')
plt.show()


# Define stopwords
stop_words = set(stopwords.words('english'))

def get_top_n_words(corpus, n=50):
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


# Get top 20 words
top_fake = get_top_n_words(fake_news)
top_true = get_top_n_words(true_news)

print("Top 30 words in fake news:")
print(top_fake)
print("\nTop 30 words in true news:")
print(top_true)

def create_wordcloud(words, title):
    print(f"Creating wordcloud: {title}")
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=300).generate_from_frequencies(dict(words))
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    plt.show()

# Create word clouds
create_wordcloud(top_fake, "Fake News")
create_wordcloud(top_true, "True News")