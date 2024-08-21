# The Impact of Deep Learning on Multilingual Toxic Data Analysis Review

The project aims to explore the effectiveness of deep learning models in detecting toxic comments across multiple languages.

## Project Overview

With the increasing use of social media, the presence of toxic content such as cyberbullying, hate speech, and abusive language has become a significant concern. This project investigates the use of deep learning models to detect and mitigate the impact of toxic comments in various languages.

### Key Highlights

- [**Multilingual Dataset**](https://www.kaggle.com/datasets/miklgr500/jigsaw-train-multilingual-coments-google-api): The study uses a multilingual dataset containing toxic comments in six different languages. 
- **Deep Learning Models**: The models used in this study include Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Long Short-Term Memory Networks (LSTM).
- **Preprocessing Techniques**: Techniques such as random undersampling, stopword removal, and lemmatization were applied to prepare the dataset.
- **Evaluation Metrics**: The models were evaluated using metrics such as accuracy, F1-score, recall, and precision. The CNN model achieved the best performance with the highest F1-score.

## Installation and Usage

To run the code in this repository, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```
2. Install the required dependencies:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn nltk spacy trnlp
   python -m spacy download es_core_news_sm
   python -m spacy download fr_core_news_sm
   python -m spacy download it_core_news_sm
   python -m spacy download pt_core_news_sm
   python -m spacy download ru_core_news_sm
   python -m spacy download tr_core_news_sm
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Main_Work_Thesis.ipynb
   ```

## Methodology

The project follows a structured methodology, starting from data gathering and preprocessing, followed by model selection, training, and evaluation.

### Data Preprocessing

- **Normalization**: Removal of special characters, numbers, and punctuation from the text.
- **Stopword Removal**: Commonly used words that do not contribute to the model's prediction were removed using the NLTK library.
- **Lemmatization**: The text was reduced to its base form to improve model performance.
- **Handling Imbalance**: Random undersampling was applied to balance the dataset.

### Models

- **CNN (Convolutional Neural Network)**: Used for its ability to capture spatial hierarchies in data, originally designed for image processing but adapted for text classification.
- **RNN (Recurrent Neural Network)**: Utilizes sequential data to maintain context over time, making it suitable for language tasks.
- **LSTM (Long Short-Term Memory)**: An advanced version of RNN, capable of learning long-term dependencies, making it effective for text classification.

## Results

The evaluation of the models showed that the CNN model performed the best with an F1-score of 0.85. The LSTM model also showed competitive performance, while the RNN model lagged slightly behind.
