# Study of RNNs - Vanilla RNN, GRU, and LSTM for Machine Translation (English to French)

## Introduction:
This project implements and compares three types of Recurrent Neural Networks (RNNs) — Vanilla RNN, GRU (Gated Recurrent Unit), and LSTM (Long Short-Term Memory) — for a machine translation task. The task involves translating English sentences into French using the Hugging Face OPUS Book dataset. The primary goal is to evaluate the performance of each RNN variant in sequence-to-sequence translation tasks, comparing translation accuracy, training time, and overall performance.

## Dataset:
### Hugging Face OPUS Book Dataset (EN-FR):
The OPUS Book dataset is an English-to-French parallel corpus available through Hugging Face’s datasets library. This dataset provides aligned sentence pairs for machine translation tasks, which makes it ideal for training and evaluating translation models.

## Preprocessing Steps:
1. Remove Punctuation: Punctuation marks are removed from both the English and French sentences to simplify the tokenization process.
2. Convert to Lowercase: All text is converted to lowercase for uniformity.
3. Tokenization: Sentences are tokenized into sequences of words, converting them into numerical tokens that represent each word.
4. Padding: Sequences are padded to ensure uniform input length, allowing for efficient batch processing.

### Data Split:
The dataset is split into training and test sets as follows:
+ 80% for training: Used to train the models.
+ 20% for testing: Used to evaluate model performance after training.

## Model Architectures:
1. Vanilla RNN
Vanilla RNN is a simple recurrent neural network where each cell has a single hidden state that is updated at each time step. However, Vanilla RNNs suffer from issues like vanishing gradients, which affect long-term dependencies.
+ Activation Function: Softmax
+ Units: 512
2. GRU (Gated Recurrent Unit)
GRU is an improvement over Vanilla RNN, addressing the vanishing gradient problem by using a gating mechanism to better retain and update information over long sequences.
+ Activation Function: Softmax
+ Units: 512
3. LSTM (Long Short-Term Memory)
LSTM networks are designed to remember information for long periods, making them well-suited for tasks like machine translation. They use a memory cell and gates to control the flow of information.
+ Activation Function: Softmax
+ Units: 512
#### Common Architecture Across Models:
All three RNN variants are implemented using the Sequential Model from Keras.

## Training Details:
### Hyperparameters
Units: 512 (for all layers)
Batch Size: 32
Epochs: 100
Optimizer: RMSProp
Learning Rate: 0.001
Loss Function: Categorical Cross-Entropy
Metrics: Accuracy
The RMSProp optimizer is used to dynamically adjust the learning rate based on recent changes in the model's weights, ensuring stable convergence.

### Loss Function
Categorical cross-entropy is used as the loss function, which is appropriate for the multi-class word prediction task in machine translation.

## Training Process:
1. Data Preprocessing: English and French sentences are cleaned, tokenized, and padded to uniform lengths.
2. Model Compilation: The Vanilla RNN, GRU, and LSTM models are compiled using the RMSProp optimizer, with accuracy as the evaluation metric.
3. Training: Each model is trained for 10 epochs on the training data with a batch size of 128, using early stopping to prevent overfitting.
4. Evaluation: After training, the models are evaluated on the test set to measure their accuracy and translation performance.

## Results:
After training, the models' performance is evaluated in terms of accuracy and translation quality. The results demonstrate how well each architecture performs for the machine translation task:
+ Vanilla RNN: Baseline performance, struggles with long sequences.
+ GRU: Improved performance over Vanilla RNN, better at capturing long-term dependencies.
+ LSTM: Best overall performance, excels in learning longer sequences and maintaining context.

## Potential Improvements:
1. Attention Mechanism: Implementing attention could improve translation quality by allowing the model to focus on relevant parts of the input sequence.
2. Beam Search: Beam search during decoding could result in more accurate translations compared to greedy search.
3. Hyperparameter Tuning: Experimenting with different hyperparameters such as learning rate, batch size, and number of units could further improve model performance.

## Acknowledgments:
Thanks to the Hugging Face team for providing easy access to datasets and models.

## Contributors:  
[Pravalika Arunkumar](https://github.com/pravalikaarunkumar)
