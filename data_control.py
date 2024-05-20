import pandas as pd

SEED = 10

# Load the WELFake_Dataset
df_WELFake = pd.read_csv('WELFake_Dataset.csv')
df_WELFake = df_WELFake.sample(frac=0.5, random_state=SEED) # Take a random sample of half the data

# Save the new data set
df_WELFake.to_csv('fake_true.csv', index=False)

# Load the True.csv and Fake.csv
df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

# Take a random sample of half the data
df_true = df_true.sample(frac=0.5, random_state=SEED)
df_fake = df_fake.sample(frac=0.5, random_state=SEED)

# Reorder the columns and add the labels
df_true = df_true[['subject', 'title', 'text', 'date']]
df_fake = df_fake[['subject', 'title', 'text', 'date']]
df_true['label'] = 1
df_fake['label'] = 0

# Add the data to the WELFake dataset
df_WELFake = pd.concat([df_WELFake, df_true, df_fake])

# Load the fake_or_real_news dataset
df_fake_or_real = pd.read_csv('fake_or_real_news.csv')

# Take a random sample of half the data
df_fake_or_real = df_fake_or_real.sample(frac=0.5, random_state=SEED)

# Rename the label to be consistent with the other data
df_fake_or_real['label'] = df_fake_or_real['label'].replace({'TRUE': 0, 'FAKE': 1})

# Add the data to the WELFake dataset
df_WELFake = pd.concat([df_WELFake, df_fake_or_real])

# Save the new dataset
df_WELFake.to_csv('WELFake_Dataset.csv', index=False)