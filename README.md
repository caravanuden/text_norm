## Text Normalization with a LSTM

Text normalization is the process of transforming text into a single canonical form that it might not have had before. In class, we've used text normalization for storing and searching information in a canonical form; ie, in the question answering assignment we converted each token to lowercase, removed punctuation, and stemmed to improve the semantic meanings stored in our word frequency vectors. Here, I extend this approach to address text-to-speech (TTS) normalization- numbers, dates, acronyms, and abbreviations are non-standard "words" in text that need to be pronounced differently in speech depending on context. For example, "123" could be pronounced "one two three" if you're counting, and "one hundred twenty-three" or even "one twenty-three" if you're referring to the number 123 or reading off an address.

My project is based on the [Google Text Normalization Challenge](https://www.kaggle.com/google-nlu/text-normalization) on Kaggle. Given a large corpus of written text aligned to its normalized spoken form, I will train an RNN to learn the correct normalization function.

### Data

The Kaggle dataset is a dataset of general text where the normalizations were generated using an existing text normalization component of a TTS system. Lines with "" in two columns are the end of sentence marker. Otherwise, there are three columns, the first of which is the "semiotic class" (Taylor, 2009), the second is the input token, and the third is the normalized output. All text is from Wikipedia. All data were extracted on 2016/04/08, and run through the Google Kestrel TTS text normalization system. The full dataset has 1 million training examples, but I am going to train my model on 500,000 training tokens (half of the full dataset) due to time and resource constraints. I will test my model on the full 100,000 test tokens.

### Linguistic intuition

LSTMs share clear parallels with high-level human language processing. Humans process and comprehend sentences within the structure of semantic and syntactical contexts. As you read this essay, you understand each word based on your understanding of previous words; you don’t throw everything away and start thinking from scratch again. Your thoughts have persistence.

Traditional neural networks can’t do this, and it seems like a major shortcoming. For example, imagine you want to classify what word will come next in a sentence. It’s unclear how a traditional neural network could use its reasoning about previous words in the sentence to inform later ones.

Recurrent neural networks (RNNs) address this issue. RNNs are networks with chain-like loops in them, which allows information to persist. This chain-like nature reveals that recurrent neural networks are intimately related to sequences and lists—they’re the natural architecture of neural network to use for such data. One of the appeals of RNNs is that they are able to connect previous (and future, for bidirectional RNNs) context information to the present task. If RNNs could do this, they’d be extremely useful. But can they? It depends.

Sometimes, we only need to look at recent information to perform the present task. In such cases, where the gap between the relevant information and the place that it’s needed is small, RNNs can learn to use the past information. But there are also cases where we need more context. Consider trying to predict the last word in the text “I grew up in France… I speak fluent French.” Recent information suggests that the next word is probably the name of a language, but if we want to narrow down which language, we need the context of France, from further back. It’s entirely possible for the gap between the relevant information and the point where it is needed to become very large.

Unfortunately, as that gap grows, RNNs become unable to learn to connect the information. In theory, RNNs are absolutely capable of handling such “long-term dependencies.” Sadly, in practice, RNNs don’t seem to be able to learn them and struggle with these long-term dependencies. on the other hand, Long Short Term Memory networks (LSTMs) are a special variant of RNN that are capable of learning long-term dependencies.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn! All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer. LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, the repeating module in an LSTM contains four interacting layers.

The key to LSTMs is the cell state, which runs straight down the entire persisting chain of the LSTM layer with only some minor linear interactions. It’s very easy for information to just flow along it unchanged. The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates. Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation. The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!” An LSTM has three of these gates, to protect and control the cell state.

Because LSTMs are able to utilize surrounding context in a long-term fashion, they are ideal for modeling NLP tasks. However, text normalization presents a few unique challenges for this approach.

### Unique challenges for text normalization with LSTMs

One issue for applying LSTMs to text normalization is that the set of interesting cases in text normalization is usually very sparse. Most tokens map to themselves, and while it is certainly important to get that right, it’s a relatively trivial case. What matters most in text normalization systems are the interesting cases—the numbers, times, dates, measure expressions, currency amounts, etc—that require special treatment.

Additionally, the requirements on accuracy are very stringent: an English TTS system had better read $123 as one hundred twenty-three dollars, but not as dollars one hundred twenty-three, one two three dollars, twelve three dollars, or even one hundred twenty-three pounds. Indeed, as we shall see below, if one is allowed to count the vast majority of cases where the right answer is to leave the input token alone, some of our RNNs already perform very well. The problem is that they tend to mess up with various semiotic classes in ways that would make them unusable for any real application, since one could never be quite sure for a new example that the system would not read it completely wrongly. As we will see below, the neural models occasionally read things like, £900 as nine hundred Euros — something that state-ofthe-art hand-built text normalization systems would never do, brittle though such systems may be. The occasional comparable error in an MT system would be bad, but it would not contribute much to a degradation of the system’s BLEU score. Such a misreading in a TTS system would be something that people would immediately notice (or, worse, not notice if they could not see the text), and would stand out precisely because a TTS system ought to get such examples right.

Additionally, though the test data were, of course, taken from a different portion of the Wikipedia text than the training data, a huge percentage of the individual tokens of the test data are likely to be found in the training set. This in itself is not surprising, but it
Raises the concern that the RNN models are just memorizing their results without doing much generalization. I’ll come back to this when analyzing the model performance.

### Model

Here, I model the whole text normalization task as one where the model map a sequence of input characters to a sequence of output words.  For the input, the string must be in terms of characters, since for a string like 123, one needs to see the individual digits in the sequence to know how to read it. On the other hand, the normalized text output, which is dependent on its surrounding sentence context, is most appropriately represented as words. However, since I need to treat the input as a sequence of characters the input layer would need to be rather large in order to cover sentences of reasonable length. It's reasonable, then, to take a different approach and place each token in a window of 3 words to the left and 3 to the right, marking the to-be-normalized token with a distinctive begin and end tag <norm> ... </norm>. Here, the token "17" in the context "My little brother turned 17 years old in April" would be represented as "little brother turned <norm> 17 </norm> years old in" in the input, which then would map to "seventeen" in the output. In this way, I can limit the number of input and output nodes to something reasonable.

 Text normalization is a sequence-to-sequence (seq2seq) problem- here, I am predicting a normalized text sequence from an input sequence. This seq2seq approach (Sutskever et al., 2014, Cho et al., 2014) underlies many NLP applications in deep learning, such as neural machine translation, speech recognition, and text summarization. My model (based on Chan et al, 2016) has two components: an encoder and a decoder. This encoder-decoder architecture for recurrent neural networks is an effective and standard approach for sequence-to-sequence (seq2seq) prediction in general. Specifically, this encoder-decoder system first reads the input using an encoder to build a high-dimensional "thought" vector that represents the sentence meaning; a decoder, then, processes the "thought" vector to generate the translated (normalized) text. The first component, the encoder, is a three-layer RNN encoder that transforms the input character sequence into a high level feature representation. The second component, the decoder, is a two-layer RNN decoder that attends to the high level features and spells out the normalized text one token at a time. The key benefit of this approach is the ability to train a single end-to-end model directly on source and target sequences.

Here is a quick summary of my model approach and architecture:

+ Character level input sequence, and word level output sequence
+ Input character vocabulary of 250 distinct characters. Output word vocabulary of 346 (the number of unique) distinct words.
+ Add a context window of 3 words to the left and right with a distinctive tag to separately identity the key token. I do this to manage the input sequence length reasonably.
+ Input sequence zero padding to a maximum of length 60. Output sequence padding to a maximum of length 20. I do this to create fixed-length sequences.
+ Model architecture with two components: an encoder and a decoder.
  - 256 hidden units in each layer of the encoder and decoder
  - Three bidirectional LSTM layers in the encoder
  - Two LSTM layers in the decoder

### Analysis of performance

### Future work

See my Jupyter notebook for the model here.
