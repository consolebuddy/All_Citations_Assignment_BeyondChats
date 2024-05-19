import torch
import spacy
import nltk
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import regex as re

nltk.download('punkt')

class CitationFinder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def remove_special_characters(self, text):
        pattern = r'[^a-zA-Z0-9 ]'
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text

    def remove_links(self, text):
        url_pattern = r'https?://\S+|www\.\S+'
        cleaned_text = re.sub(url_pattern, '', text)
        return cleaned_text

    def lemmatize_words(self, words):
        doc = self.nlp(' '.join(words))
        lemmatized_words = [token.lemma_ for token in doc]
        return lemmatized_words

    def preprocess_text(self, text):
        # text = self.remove_links(text)
        text = self.remove_special_characters(text)
        text = ' '.join(text.split())
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.lower() not in self.stop_words]
        text = ' '.join(tokens)
        return text

    def embed_text(self, text):
        tokens = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**tokens)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Get the embeddings of [CLS] token
        return embeddings

    def cosine_similarity(self, embedding1, embedding2):
        return torch.nn.functional.cosine_similarity(embedding1, embedding2).item()

    def get_similarity(self, text1, text2):
        embedding1 = self.embed_text(text1)
        embedding2 = self.embed_text(text2)
        return self.cosine_similarity(embedding1, embedding2)

    def extract_keywords(self, response):
        doc = self.nlp(response)
        keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        keywords = self.lemmatize_words(keywords)
        seen = set()
        keywords = [x for x in keywords if not (x in seen or seen.add(x))]
        return keywords

    def clean_citations(self, citations):
        citations = list({item['id']: item for item in citations}.values())
        citations = list({item['context'].lower(): item for item in citations}.values())
        citations = [{"id": citation["id"], "link": citation["link"]} for citation in citations]
        return citations


    def find_citations(self, response, sources, thres=0.8):
        citations = []
        for r in response.split('.'):
            r = self.preprocess_text(r)
            keywords = self.extract_keywords(r)

            for source in sources:
                context = source["context"]
                if isinstance(context, list):
                    context = ' '.join(context)
                context = self.preprocess_text(context)
                similarity_score = self.get_similarity(r, context)

                if any(keyword in context.lower() for keyword in keywords) and similarity_score > thres:
                    citations.append({"id": source["id"], "context": source["context"], "link": source["link"]})

        return self.clean_citations(citations)