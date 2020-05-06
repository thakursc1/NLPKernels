

import re  # Regular Expressions

import matplotlib.pyplot as plt
# | ModelName     | Accuracy Reported|
# | ------------- |:-------------:|
# | Simple Bidirectional Lstm | F1 Score: |
# | Bert | F1 Score: |
# | Roberta | F1 Score: |
#
# 3. Conclusion
#
# #%%
#
# !pip install tokenizers
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K

plt.style.use('ggplot')

config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# Calculate F1:
def f1_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# List Input Directories
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.

# %% md
#
# ### Data Preprocessing
# 1.1 Remove Unwanted data from the tweets including urls, user names incorrect words

# %%

# Get rid of urls, hashtags, @usernames, emojis via regex
# References: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python

# Emojis are a good representation of emotions but for the case of simplicity lets ignore them for now

emoji_pattern = emoji_pattern = re.compile("["
                                           u"\U0001F600-\U0001F64F"  # emoticons
                                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                           "]+", flags=re.UNICODE)
misc_pattern = re.compile('<.*?>|^http?:\/\/.*[\r\n]*|#|@[^\s]+|http[s]?[^\s]+')
numeric_pattern = re.compile("[0-9]+")
punctuations_pattern = re.compile('[^\w\s]')


# html_pattern '<.*?>'
# url_pattern '^https?:\/\/.*[\r\n]*'
# hash_tags '#'
# username pattern ''@[^\s]+'
# Remove Numbers and complressed urls as such httpssampleurlstoremove using:
# [0-9]+|http[s]?[^\s]+

# def clean_tweet(tweet):
#     tweet = re.sub(emoji_pattern, '', tweet)
#     tweet = re.sub(misc_pattern, '', tweet)
#     tweet = re.sub(punctuations_pattern, '', tweet)
#     tweet = tweet.lower()
#     return remove_stopwords(tweet.strip())

# Since Bert Tokenizer is already trained on many of the emojis we only remove urls
def clean_tweet(tweet):
    tweet = re.sub(misc_pattern, '', tweet)
    tweet = re.sub(punctuations_pattern, '', tweet)
    tweet = tweet.lower()
    return tweet.strip()


example = " <h1> https://bit.lu/3849yedjk Our Deeds are the Reason @remove_me, of this #earthquake May ALLAH Forgive us all 😔😔, "
print(clean_tweet(example))  # no emoji

# %%

# Read and Preprocess the dataset
train_df = pd.read_csv(r"kaggle/input/train.csv")
test_df = pd.read_csv(r"kaggle/input/test.csv")
print(train_df.shape, test_df.shape)

# Preprocessing
train_df['text_processed'] = train_df['text'].apply(lambda x: clean_tweet(x))
test_df['text_processed'] = test_df['text'].apply(lambda x: clean_tweet(x))
train_df['keyword'] = train_df['keyword'].apply(lambda x: " ".join(str(x).strip().split("%20")))
test_df['keyword'] = test_df['keyword'].apply(lambda x: " ".join(str(x).strip().split("%20")))

# Looks Like we happen to have a lot of nans in the keywords
train_df.loc[train_df['keyword'] == 'nan', 'keyword'] = np.nan
test_df.loc[train_df['keyword'] == 'nan', 'keyword'] = np.nan

# Check class imbalance
# Looks like is slightly imbalanced, we can solve this issue by sampling accordingly
print(train_df["target"].value_counts())

# Dropping location as it unavailable for more the 30% dataset in train and 14% in test
train_df.drop("location", axis=1, inplace=True)
test_df.drop("location", axis=1, inplace=True)

# Imputing the missing key words to unknown
train_df["keyword"].fillna("unknown", inplace=True)

# %%

# Print Data Quality Stats
train_df.isnull().sum() / train_df.shape[0] * 100

# %%

test_df.isnull().sum() / train_df.shape[0] * 100

# %%

# Peeking at the action tweets
list(train_df[['text', 'text_processed', "target"]].sample(n=5).values)

# #%% md
#
# 1.2. Remove Stopwords and tokenize using HuggingFace BertWordPiece Tokenizer
#
# #%%

# Train a custom BertWordPieceTokenizer which takes into account most of the tokens due to word piece stratergy
# minimizing number of unknowns

# %% md
#
# 3. What a disaster tweet looks like ?

# %%

# Word Count Distributions
sns.distplot(train_df['text_processed'].apply(lambda x: len(x)), label='train')
sns.distplot(test_df['text_processed'].apply(lambda x: len(x)), label='test')
plt.legend()
# This helps to set our max tweet length to 150
MAX_TWEET_LEN = train_df['text_processed'].apply(lambda x: len(x)).max()
# print("Max Tweet length ", MAX_TWEET_LEN)
# # Enable Padding and Truncation at Max Length
# tokenizer.enable_padding(max_length=MAX_TWEET_LEN)
# tokenizer.enable_truncation(max_length=MAX_TWEET_LEN)
# print("Tokenized Example: ", tokenizer.encode(clean_tweet(example)).tokens)
# print("Vocab Size: ", tokenizer.get_vocab_size())

# %%

# Most common words in the dataset
disaster_tweet_words = [word for i in
                        train_df.loc[train_df['target'] == 1, 'text_processed'].apply(lambda x: x.split()).to_list() for
                        word in i]
normal_tweet_words = [word for i in
                      train_df.loc[train_df['target'] == 0, 'text_processed'].apply(lambda x: x.split()).to_list() for
                      word in i]
disaster_tweet_words_df = pd.Series(disaster_tweet_words)
normal_tweet_words_df = pd.Series(normal_tweet_words)

# Top Most common keywords
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
disaster_tweet_words_df.value_counts()[:20].plot(kind='barh', ax=ax1, color='green')
normal_tweet_words_df.value_counts()[:20].plot(kind='barh', ax=ax2)
fig.suptitle("Most Common words in tweets by target")
# Top Most common keywords
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
# Keywords by Target
train_df[train_df['target'] == 1]['keyword'].value_counts()[:10].plot(kind='barh', ax=ax1, color='green')
train_df[train_df['target'] == 0]['keyword'].value_counts()[:10].plot(kind='barh', ax=ax2)
fig.suptitle("Most Common keywords by target")
# Hack: Instead of using keywords as a separate feature, lets add it towards the end of our tweets
train_df['text_processed'] = train_df['text_processed'] + " " + train_df['keyword']
test_df['text_processed'] = test_df['text_processed'] + " " + test_df['keyword']

#
# 2. Try BERT (using HuggingFace Implementation)
# ![BERT](http://www.mccormickml.com/assets/BERT/padding_and_mask.png)
# pic credits: [https://mccormickml.com/](https://mccormickml.com/)

from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

print('Example :', clean_tweet(example))

# Apply the tokenizer to the input text, treating them as a text-pair.
print(tokenizer.tokenize(clean_tweet(example)))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(clean_tweet(example))))


# Used https://github.com/huggingface/transformers/blob/a638e986f45b338c86482e1c13e045c06cfeccad/src/transformers/data/processors/glue.py#L34
# for references on making a tensor flow dataset

def get_tokenized_data(df, train=True):
    train_sequences = df['text_processed'].to_list()

    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_sequence_ids = []
    attention_masks = []

    for sequence in train_sequences:
        # Add the encoded sentence to the list.
        input_sequence_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    sequences = tf.stack(input_sequence_ids)
    attention_masks = tf.stack(attention_masks)
    if train:
        labels = tf.convert_to_tensor(train_df['target'], dtype=tf.int32)
        return sequences, attention_masks, labels
    return sequences, attention_masks


from transformers import TFBertForSequenceClassification

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.


train_size = int(0.9 * (train_df.shape[0]))
val_size = train_df.shape[0] - train_size

# Training Data
train_dataset = tokenizer.batch_encode_plus(
    train_df.iloc[:train_size]['text_processed'].values,
    add_special_tokens=True,
    max_length=MAX_TWEET_LEN,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='tf'
)
train_labels = train_df.iloc[:train_size]['target'].values.reshape((-1, 1))

# Validation Data
val_dataset = tokenizer.batch_encode_plus(
    train_df.iloc[train_size:]['text_processed'].values,
    add_special_tokens=True,
    max_length=MAX_TWEET_LEN,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='tf'
)
val_labels = train_df.iloc[train_size:]['target'].values.reshape((-1, 1))

# Testing Data for making predictions
test_dataset = tokenizer.batch_encode_plus(
    test_df['text_processed'].values,
    add_special_tokens=True,
    max_length=MAX_TWEET_LEN,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='tf'
)


def create_model():
    bert_model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased")

    input_ids = tf.keras.layers.Input((MAX_TWEET_LEN,), dtype=tf.int32, name='input_ids')
    token_type_ids = tf.keras.layers.Input((MAX_TWEET_LEN,), dtype=tf.int32, name='token_type_ids')
    attention_mask = tf.keras.layers.Input((MAX_TWEET_LEN,), dtype=tf.int32, name='attention_mask')

    # Use pooled_output(hidden states of [CLS]) as sentence level embedding
    pooled_output = \
        bert_model({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})[0]
    x = tf.keras.layers.Dropout(rate=0.1)(pooled_output)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(
        inputs={'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, outputs=x)
    return model


EPOCHS = 2
BATCH_SIZE = 2
model = create_model()

print(model.summary())

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
opt = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['Precision', 'Recall', f1_score])

# Train and evaluate using tf.keras.Model.fit()
history = model.fit(
    x=train_dataset,
    y=train_labels,
    validation_data=(val_dataset, val_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Load the TensorFlow model in PyTorch for inspection
model.save_pretrained('./save/')

from sklearn.metrics import classification_report

test_df['probs'] = model.predict(test_dataset)
test_df['target'] = (test_df['probs'] > 0.5).astype(int)
test_df[['id', 'target']].to_csv('BertSubmission.csv', index=False)
print(classification_report(test_df['probs'], test_df['']))
