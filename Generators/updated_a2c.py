import os
import random
import csv
import xml.etree.ElementTree as ET
import spacy
import torch
import traceback
import logging
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BartTokenizer,
    BartForConditionalGeneration,
)
from datasets import Dataset
from bs4 import BeautifulSoup
import openai

# Download NLTK data required for sentence tokenization
nltk.download('punkt')

# ==============================
# 1. Configuration and Setup
# ==============================

# Configure logging to capture training progress and errors
logging.basicConfig(
    filename='training.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Securely load your OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize Spacy for NLP tasks (e.g., extracting semantic roles)
try:
    nlp = spacy.load('en_core_web_sm')  # This is the NER extraction model from Spacy
except OSError:
    # Download the model if not already present
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Determine the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Set maximum tokens per request for OpenAI API
MAX_TOKENS_PER_REQUEST = 2000  # Adjust as needed
MAX_WORDS_FINAL_SUMMARY = 250  # Adjust as needed

# Initialize BART tokenizer and model for semantic compression
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)

# ==============================
# 1.1. HumanlikeSummarizer Class Integration
# ==============================

class HumanlikeSummarizer:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.thought_patterns = [
            "This suggests that", "Interestingly,", "It's worth noting that",
            "Upon closer examination,", "This raises questions about"
        ]
        
    def analyze_source_style(self, text):
        """
        Analyze the writing style of source text to maintain consistency.
        """
        doc = self.nlp(text)
        style_metrics = {
            'sentence_length': [],
            'vocabulary_level': defaultdict(int),
            'transition_patterns': [],
            'punctuation_patterns': defaultdict(int)
        }
        
        # Analyze sentence lengths and patterns
        for sent in doc.sents:
            style_metrics['sentence_length'].append(len(sent))
            
            # Track sophisticated vocabulary
            for token in sent:
                if token.is_alpha and len(token.text) > 7:
                    style_metrics['vocabulary_level'][token.text.lower()] += 1
                    
            # Track transition words at sentence starts
            first_token = next(iter(sent)).text.lower()
            if first_token in ['however', 'moreover', 'therefore', 'consequently']:
                style_metrics['transition_patterns'].append(first_token)
                
        return style_metrics
    
    def introduce_cognitive_load_variations(self, text):
        """
        Add natural variations in complexity to mimic human cognitive load patterns.
        """
        sentences = sent_tokenize(text)
        modified_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Add occasional self-corrections
            if random.random() < 0.1:
                sentence = self._add_self_correction(sentence)
                
            # Vary sentence complexity based on position
            if i > 0 and random.random() < 0.15:
                sentence = self._simplify_sentence(sentence)
            elif random.random() < 0.2:
                sentence = self._add_hedge_words(sentence)
                
            modified_sentences.append(sentence)
            
        return ' '.join(modified_sentences)
    
    def _add_self_correction(self, sentence):
        """Add natural self-corrections that humans make while writing."""
        correction_patterns = [
            (r'\b(is)\b', r'was— no, is'),
            (r'\b(will)\b', r'would— actually, will'),
            (r'\b(many)\b', r'several— or rather, many')
        ]
        
        if random.random() < 0.3:
            pattern, replacement = random.choice(correction_patterns)
            sentence = re.sub(pattern, replacement, sentence, count=1)
        return sentence

    def _simplify_sentence(self, sentence):
        """Simplify complex sentences."""
        doc = self.nlp(sentence)
        simplified_sentence = ' '.join([token.text for token in doc if not token.is_stop])
        return simplified_sentence

    def _add_hedge_words(self, sentence):
        """Add hedge words to make statements less absolute."""
        hedge_words = ['perhaps', 'possibly', 'might', 'could']
        words = sentence.split()
        if words:
            insert_position = random.randint(0, len(words) - 1)
            words.insert(insert_position, random.choice(hedge_words))
            return ' '.join(words)
        return sentence

    def add_perspective_shifts(self, text):
        """
        Introduce subtle perspective shifts that occur naturally in human writing.
        """
        sentences = sent_tokenize(text)
        for i in range(len(sentences)):
            if random.random() < 0.15:
                perspective = random.choice([
                    "From this perspective,",
                    "Looking at it differently,",
                    "One could argue that",
                    "It's important to consider that"
                ])
                sentences[i] = f"{perspective} {sentences[i]}"
        
        return ' '.join(sentences)

    def introduce_emphasis_patterns(self, text):
        """
        Add natural emphasis patterns found in human writing.
        """
        sentences = sent_tokenize(text)
        modified_sentences = []
        
        for sentence in sentences:
            # Add emphasis through repetition or reinforcement
            if random.random() < 0.1:
                words = sentence.split()
                for i, word in enumerate(words):
                    if len(word) > 5 and random.random() < 0.15:
                        words[i] = f"{word} — yes, {word}"
                sentence = ' '.join(words)
            
            modified_sentences.append(sentence)
            
        return ' '.join(modified_sentences)

    def add_rhetorical_devices(self, text):
        """
        Incorporate natural rhetorical devices used in human writing.
        """
        sentences = sent_tokenize(text)
        modified_sentences = []
        
        for i, sentence in enumerate(sentences):
            if random.random() < 0.2:
                rhetorical_device = random.choice([
                    self._add_analogy,
                    self._add_rhetorical_question,
                    self._add_parallel_structure
                ])
                sentence = rhetorical_device(sentence)
            modified_sentences.append(sentence)
            
        return ' '.join(modified_sentences)

    def _add_analogy(self, sentence):
        """Add relevant analogies to complex concepts."""
        analogies = [
            "much like how",
            "similar to",
            "just as",
            "comparable to"
        ]
        if len(sentence) > 50:  # Only add analogies to longer, complex sentences
            return f"{sentence} — {random.choice(analogies)} {self._generate_simple_analogy()}"
        return sentence

    def _generate_simple_analogy(self):
        """Generate simple, relatable analogies."""
        analogies = [
            "how water flows downhill",
            "pieces fitting in a puzzle",
            "branches growing from a tree",
            "colors mixing in a painting"
        ]
        return random.choice(analogies)

    def _add_rhetorical_question(self, sentence):
        """Add a rhetorical question related to the sentence."""
        return f"{sentence} Isn't that intriguing?"

    def _add_parallel_structure(self, sentence):
        """Add parallel structure to emphasize a point."""
        return f"{sentence} We must consider, we must act, we must change."

    def add_temporal_markers(self, text):
        """
        Add natural time-based transitions and markers.
        """
        sentences = sent_tokenize(text)
        temporal_markers = [
            "Initially,", "Subsequently,", "Later,", "Finally,",
            "At first,", "Eventually,", "In the end,"
        ]
        
        # Add temporal markers strategically
        if len(sentences) > 3:
            sentences[0] = f"{random.choice(temporal_markers)} {sentences[0]}"
            mid_point = len(sentences) // 2
            sentences[mid_point] = f"{random.choice(temporal_markers)} {sentences[mid_point]}"
            
        return ' '.join(sentences)

    def implement_natural_digression(self, text):
        """
        Add occasional natural digressions that humans make while writing.
        """
        sentences = sent_tokenize(text)
        modified_text = []
        
        for i, sentence in enumerate(sentences):
            modified_text.append(sentence)
            
            # Add occasional relevant digressions
            if random.random() < 0.15 and i < len(sentences) - 1:
                digression = self._generate_relevant_digression(sentence)
                if digression:
                    modified_text.append(digression)
                
        return ' '.join(modified_text)

    def _generate_relevant_digression(self, sentence):
        """Generate contextually relevant digressions."""
        doc = self.nlp(sentence)
        key_entities = [ent.text for ent in doc.ents]
        
        if key_entities:
            entity = random.choice(key_entities)
            digression_patterns = [
                f"(Interestingly, {entity} reminds me of a related point...)",
                f"(This aspect of {entity} is particularly noteworthy...)",
                f"(It's worth briefly mentioning about {entity} that...)"
            ]
            return random.choice(digression_patterns)
        return ""

def enhance_summary_with_human_features(text, summarizer):
    """
    Apply all human-like enhancement techniques to the summary.
    """
    # Analyze source style first
    style_metrics = summarizer.analyze_source_style(text)
    
    # Apply enhancements in a natural sequence
    enhanced_text = text
    enhanced_text = summarizer.introduce_cognitive_load_variations(enhanced_text)
    enhanced_text = summarizer.add_perspective_shifts(enhanced_text)
    enhanced_text = summarizer.introduce_emphasis_patterns(enhanced_text)
    enhanced_text = summarizer.add_rhetorical_devices(enhanced_text)
    enhanced_text = summarizer.add_temporal_markers(enhanced_text)
    enhanced_text = summarizer.implement_natural_digression(enhanced_text)
    
    return enhanced_text

def integrate_human_features(summary_text, summarizer):
    """
    Integrate all human-like features into the existing summary pipeline.
    """
    enhanced_summary = enhance_summary_with_human_features(summary_text, summarizer)
    return enhanced_summary

# ==============================
# 2. Define Utility Functions
# ==============================

def parse_sgml_file(sgml_file_path):
    """
    Parse an SGML file and extract articles.
    """
    try:
        with open(sgml_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Use BeautifulSoup to parse SGML
        soup = BeautifulSoup(content, 'html.parser')

        articles = []
        for topic in soup.find_all('topic'):
            topic_id = topic.find('num').text.strip() if topic.find('num') else 'unknown'
            docs = topic.find('docs').text.strip().split() if topic.find('docs') else []
            for doc in docs:
                articles.append({'topic_id': topic_id, 'doc_id': doc})

        return articles

    except Exception as e:
        logging.error(f"Failed to parse SGML file {sgml_file_path}: {str(e)}")
        return []

def load_articles(input_directory, sgml_file):
    """
    Load articles from SGML file and corresponding structured files.
    """
    # Use sgml_file directly if it is an absolute or relative path
    if os.path.isabs(sgml_file) or sgml_file.startswith('.'):
        sgml_path = sgml_file
    else:
        sgml_path = os.path.join(input_directory, sgml_file)
    parsed_articles = parse_sgml_file(sgml_path)
    articles = []

    for article in parsed_articles:
        doc_id = article['doc_id']
        # Assuming files are named as doc_id without any extension
        article_file = f"{doc_id}"
        article_path = os.path.join(input_directory, article_file)
        
        if os.path.exists(article_path):
            try:
                with open(article_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                
                # Parse using BeautifulSoup as it can handle both HTML-like and SGML-like structures
                soup = BeautifulSoup(content, 'html.parser')

                # Extract relevant sections
                headline = soup.find('headline').get_text() if soup.find('headline') else ''
                text_content = ' '.join([p.get_text() for p in soup.find_all('p')])  # Extract all <P> content

                # Combine headline and body text
                full_text = f"{headline}\n{text_content}"
                articles.append({'topic_id': article['topic_id'], 'text': full_text})
            except Exception as e:
                logging.warning(f"Failed to read or parse {article_path}: {str(e)}")
        else:
            logging.warning(f"Article file {article_path} does not exist.")

    return articles

def estimate_tokens(text, tokenizer):
    """
    Estimate the number of tokens for the given text.
    """
    return len(tokenizer.encode(text))

def clean_text(text):
    """
    Clean the input text by removing non-ASCII characters and extra whitespace.
    """
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()     # Remove extra whitespace
    return text

def chunk_text_by_paragraphs(text, max_tokens, tokenizer):
    """
    Splits the text into smaller chunks by paragraph that fit within the token limit.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = estimate_tokens(paragraph, tokenizer)
        if current_length + paragraph_length > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
        current_chunk.append(paragraph)
        current_length += paragraph_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def extract_key_roles(text, nlp_model):
    """
    Extract key semantic roles (subjects, verbs, objects) from the text using Spacy.
    """
    doc = nlp_model(text)
    roles = {'subjects': [], 'objects': [], 'verbs': []}
    for token in doc:
        if token.dep_ == 'nsubj':
            roles['subjects'].append(token.text)
        elif token.dep_ == 'dobj':
            roles['objects'].append(token.text)
        elif token.pos_ == 'VERB':
            roles['verbs'].append(token.text)
    return roles

def summarize_with_roles(text, tokenizer, model, device, nlp_model):
    # Extract key roles as before
    roles = extract_key_roles(text, nlp_model)
    roles_text = f"Subjects: {', '.join(set(roles['subjects']))}\n"
    roles_text += f"Verbs: {', '.join(set(roles['verbs']))}\n"
    roles_text += f"Objects: {', '.join(set(roles['objects']))}"

    # Truncate the text to leave room for the prompt and generated summary
    max_input_length = 800  # Adjust as needed to fit within 1024 tokens
    text = tokenizer.decode(
        tokenizer.encode(text, max_length=max_input_length, truncation=True)
    )

    prompt = (
        "Please summarize the following text, ensuring to include the key subjects, verbs, and objects:\n\n"
        f"{roles_text}\n\nText:\n{text}"
    )

    # Tokenize the prompt with truncation
    encoding = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=1024 - 150,  # Leave space for generation
        padding=True
    ).to(device)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Generate summary using the model
    summary_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=150,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def compute_reward(summary, reward_model, reward_tokenizer, device):
    """
    Compute the reward for a given summary using the reward model.
    """
    with torch.no_grad():
        inputs = reward_tokenizer(summary, return_tensors='pt', truncation=True, max_length=512).to(device)
        outputs = reward_model(**inputs)
        # Assuming the reward model outputs logits for binary classification
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        reward = probabilities[0][1].item()  # Assuming index 1 corresponds to the positive class
    return reward

def content_filter(summary):
    """
    Check the summary for inappropriate content using OpenAI's Moderation API.
    """
    try:
        response = openai.Moderation.create(input=summary)
        results = response["results"][0]
        if results["flagged"]:
            logging.warning(f"Content flagged as inappropriate: {results['categories']}")
            return False  # Content is inappropriate
        return True  # Content is appropriate
    except Exception as e:
        logging.error(f"Content filtering failed: {str(e)}")
        return False  # Treat content as inappropriate if filtering fails

def enhance_and_trim_summary(summary, max_words, summarizer):
    """
    Enhance the summary and trim it to the maximum word count.
    """
    # Apply human-like features using the summarizer
    summary = integrate_human_features(summary, summarizer)
    words = summary.split()
    if len(words) > max_words:
        summary = ' '.join(words[:max_words]) + '.'
    return summary

def paraphrase_with_variability(summary):
    """
    Use OpenAI's GPT to paraphrase the summary, introducing stylistic variability.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a paraphrasing assistant that rephrases text to introduce stylistic variability."},
                {"role": "user", "content": f"Please paraphrase the following summary:\n\n{summary}"}
            ],
            max_tokens=250
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logging.error(f"Paraphrasing failed: {str(e)}")
        return summary  # Return original summary if paraphrasing fails

def back_translate(summary, intermediate_language="French"):
    """
    Translate the summary to an intermediate language and back to English to introduce variability.
    """
    try:
        # Translate to intermediate language
        response1 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Translate the following text to {intermediate_language}:\n\n{summary}"}
            ],
            max_tokens=500
        )
        translated_text = response1.choices[0].message['content'].strip()

        # Translate back to English
        response2 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Translate the following {intermediate_language} text back to English:\n\n{translated_text}"}
            ],
            max_tokens=500
        )
        back_translated_summary = response2.choices[0].message['content'].strip()
        return back_translated_summary
    except Exception as e:
        logging.error(f"Back-translation failed: {str(e)}")
        return summary  # Return original summary if back-translation fails

def remove_special_characters(summary):
    """
    Remove unnecessary special characters from the summary.
    """
    summary = summary.replace(';', ',').replace('+', 'and')
    # Avoid over-sanitization
    summary = summary.replace('...', '.').replace('!!', '!').replace('??', '?')
    return summary

def ai_detection_score_roberta(summary, reward_model, reward_tokenizer, device):
    """
    Use your RoBERTa model to detect if the summary is AI-generated.
    Returns a score between 0 (human-written) and 1 (AI-generated).
    """
    try:
        inputs = reward_tokenizer(summary, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = reward_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        score = probabilities[0][1].item()  # Probability of being AI-generated
        return score
    except Exception as e:
        logging.error(f"AI detection failed: {str(e)}")
        return 0.0  # Default to human-written if detection fails

def semantic_compression(text):
    """
    Compress the text semantically using a pre-trained summarization model (BART).
    """
    try:
        # Tokenize the input text and get attention mask
        encoding = bart_tokenizer(
            text,
            return_tensors='pt',
            max_length=800,
            truncation=True,
            padding=True  # Ensure consistent length if batching
        ).to(device)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Generate summary (compressed version)
        summary_ids = bart_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=4,
            length_penalty=2.0,
            max_new_tokens=150,
            min_length=40,
            no_repeat_ngram_size=3,
            pad_token_id=bart_tokenizer.pad_token_id,  # Use bart_tokenizer's pad_token_id
            early_stopping=True
        )
        
        # Decode the generated summary
        compressed_text = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return compressed_text
    except Exception as e:
        logging.error(f"Semantic compression failed: {str(e)}")
        return text  # Return original text if compression fails

def blend_summaries(summaries):
    """
    Blend multiple summaries into one coherent summary.
    """
    return ' '.join(summaries)

def iterative_summary_refinement(summary, iterations=2):
    """
    Refine the summary iteratively using OpenAI's GPT model to enhance fluency and coherence.
    """
    try:
        refined_summary = summary
        for _ in range(iterations):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that refines text to improve clarity, coherence, and fluency without changing the original meaning."},
                    {"role": "user", "content": f"Please refine the following summary:\n\n{refined_summary}"}
                ],
                max_tokens=512
            )
            refined_summary = response.choices[0].message['content'].strip()
        return refined_summary
    except Exception as e:
        logging.error(f"Iterative summary refinement failed: {str(e)}")
        return summary  # Return original summary if refinement fails

def parse_topics_file(topics_file):
    """
    Parse the topics file and return a dictionary mapping topic IDs to article filenames.
    """
    topics = {}
    try:
        with open(topics_file, 'r', encoding='utf-8') as file:
            content = file.read()

        soup = BeautifulSoup(content, 'html.parser')

        for topic in soup.find_all('topic'):
            topic_id = topic.find('num').text.strip() if topic.find('num') else 'unknown'
            docs = topic.find('docs').text.strip().split() if topic.find('docs') else []
            topics[topic_id] = docs

    except Exception as e:
        logging.error(f"Failed to parse topics file {topics_file}: {str(e)}")

    return topics

# ==============================
# 3. Define the A2C Training Loop
# ==============================

def train_a2c_model(policy_model, value_model, train_dataset, reward_model, reward_tokenizer, tokenizer, device, nlp_model):
    NUM_EPOCHS = 3
    MAX_LENGTH = 512  # Adjust as needed
    gamma = 0.99  # Discount factor for rewards

    # Define optimizers for policy and value models
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=5e-6)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=1e-5)

    for epoch in range(NUM_EPOCHS):
        logging.info(f"Starting epoch {epoch+1}/{NUM_EPOCHS}")

        for idx, batch in enumerate(train_dataset):
            try:
                input_text = batch['text']
                input_text = clean_text(input_text)

                # Tokenize input text with attention mask
                encoding = tokenizer(
                    input_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=True
                ).to(device)
                input_ids = encoding['input_ids']
                attention_mask = encoding ['attention_mask']
                # Generate summary
                summary = summarize_with_roles(input_text, tokenizer, policy_model, device, nlp_model)

                # Compute reward using the reward model
                reward_value = compute_reward(summary, reward_model, reward_tokenizer, device)
                reward = torch.tensor(reward_value, dtype=torch.float32, device=device)
                reward = reward.view(1)  # Ensure shape is [1]

                # Tokenize summary with attention mask
                summary_encoding = tokenizer(
                    summary,
                    return_tensors='pt',
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=True
                ).to(device)
                summary_ids = summary_encoding['input_ids']
                summary_attention_mask = summary_encoding['attention_mask']

                # Prepare inputs for value model
                value_input_ids = torch.cat([input_ids, summary_ids], dim=1)
                value_attention_mask = torch.cat([attention_mask, summary_attention_mask], dim=1)

                # Estimate value using the value model (critic)
                value_outputs = value_model(
                    input_ids=value_input_ids,
                    attention_mask=value_attention_mask
                )
                value = value_outputs.logits[:, -1].mean()
                value = value.unsqueeze(0)  # Ensure shape is [1]

                # Compute advantage
                advantage = reward - value.detach()

                # Compute policy loss
                policy_outputs = policy_model(
                    input_ids=value_input_ids,
                    attention_mask=value_attention_mask
                )
                log_probs = torch.nn.functional.log_softmax(policy_outputs.logits[:, -1, :], dim=-1)
                action = summary_ids[:, -1]
                action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
                policy_loss = -action_log_prob * advantage

                # Backpropagate policy loss
                policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                policy_optimizer.step()

                # Compute value loss
                value_loss = torch.nn.functional.mse_loss(value, reward)

                # Backpropagate value loss
                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)
                value_optimizer.step()

                logging.info(f"Epoch {epoch+1}, Step {idx+1}: Reward={reward.item():.4f}, Policy Loss={policy_loss.item():.4f}, Value Loss={value_loss.item():.4f}")

            except Exception as e:
                logging.error(f"Error during training loop at Epoch {epoch+1}, Step {idx+1}: {str(e)}")
                logging.error(traceback.format_exc())
                continue  # Skip to the next batch

        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} completed.")

    # Optionally save the models
    policy_model.save_pretrained('./a2c_policy_model')
    value_model.save_pretrained('./a2c_value_model')

# ==============================
# 4. Define the RL Summarization Function
# ==============================

def rl_summarize(text, policy_model, value_model, tokenizer, device, nlp_model, reward_model, reward_tokenizer, perform_a2c_step=True):
    """
    Generate a summary using the policy model, compute the reward, and optionally perform an A2C update.
    """
    try:
        text = clean_text(text)
        max_input_length = 800
        input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_input_length).to(device)

        # Generate summary
        summary = summarize_with_roles(text, tokenizer, policy_model, device, nlp_model)

        # Compute reward
        reward_value = compute_reward(summary, reward_model, reward_tokenizer, device)
        reward = torch.tensor([reward_value], dtype=torch.float32, device=device)

        if perform_a2c_step:
            # Prepare inputs for value model
            summary_ids = tokenizer.encode(summary, return_tensors='pt', truncation=True, max_length=max_input_length).to(device)
            value_input_ids = torch.cat([input_ids, summary_ids], dim=1)

            # Estimate value of the state using the value model (critic)
            value_outputs = value_model(value_input_ids)
            value = value_outputs.logits[:, -1, :].mean()

            # Compute advantage
            advantage = reward - value.detach()

            # Compute policy loss
            log_probs = torch.nn.functional.log_softmax(policy_model(value_input_ids).logits[:, -1, :], dim=-1)
            action = summary_ids[:, -1]
            action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
            policy_loss = -action_log_prob * advantage

            # Backpropagate policy loss
            policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=5e-6)
            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            policy_optimizer.step()

            # Compute value loss
            value_loss = torch.nn.functional.mse_loss(value, reward)

            # Backpropagate value loss
            value_optimizer = torch.optim.Adam(value_model.parameters(), lr=1e-5)
            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)
            value_optimizer.step()

        return summary, reward_value

    except Exception as e:
        logging.error(f"Failed to generate summary: {str(e)}")
        logging.error(traceback.format_exc())
        return None, None

# ==============================
# 5. Define the Processing Function
# ==============================

def process_and_save_summaries(input_directory, output_directory, results_file, topics_file, policy_model, value_model, tokenizer, device, nlp_model, reward_model, reward_tokenizer):
    """
    Process all articles by topics, generate summaries with human-like features,
    save summaries in another directory, and save detection results to a CSV file.
    """
    results = []
    root_element = ET.Element("GeneratorResults", teamName="PV-Credit")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    topics = parse_topics_file(topics_file)

    # Initialize the summarizer with the NLP model
    summarizer = HumanlikeSummarizer(nlp_model)

    for topic_id, article_filenames in topics.items():
        article_contents = []

        for filename in article_filenames:
            article_path = os.path.join(input_directory, filename)
            if os.path.exists(article_path):
                with open(article_path, 'r', encoding='utf-8') as file:
                    article_contents.append(file.read())
            else:
                print(f"File not found: {article_path}")
                logging.warning(f"File not found: {article_path}")

        if not article_contents:
            print(f"No articles found for topic {topic_id}.")
            logging.warning(f"No articles found for topic {topic_id}.")
            continue

        combined_content = "\n\n".join(article_contents)
        combined_content = semantic_compression(combined_content)

        # Chunk the content if necessary
        chunks = chunk_text_by_paragraphs(combined_content, MAX_TOKENS_PER_REQUEST, tokenizer) if estimate_tokens(combined_content, tokenizer) > MAX_TOKENS_PER_REQUEST else [combined_content]

        summaries = []
        for chunk in chunks:
            # Generate summary with RL fine-tuning
            summary, reward = rl_summarize(
                chunk,
                policy_model=policy_model,
                value_model=value_model,
                tokenizer=tokenizer,
                device=device,
                nlp_model=nlp_model,
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                perform_a2c_step=False  # Disable A2C step during processing
            )
            if summary is not None:
                # Enhance and refine summary using the summarizer
                summary = enhance_and_trim_summary(summary, MAX_WORDS_FINAL_SUMMARY, summarizer)
                summary = remove_special_characters(summary)
                summary = paraphrase_with_variability(summary)
                summary = back_translate(summary)
                summary = iterative_summary_refinement(summary)

                summaries.append(summary)
                logging.info(f"Processed chunk for topic {topic_id}")
            else:
                logging.warning(f"Failed to generate summary for chunk in topic {topic_id}")

        combined_summary = blend_summaries(summaries)
        final_summary = combined_summary

        # Check summary with your RoBERTa model
        score = ai_detection_score_roberta(final_summary, reward_model, reward_tokenizer, device)
        detection_result = "Human" if score < 0.5 else "AI"

        # Content filtering
        if not content_filter(final_summary):
            print(f"Summary for topic {topic_id} contains inappropriate content and will not be saved.")
            logging.warning(f"Summary for topic {topic_id} contains inappropriate content and will not be saved.")
            continue

        results.append([topic_id, detection_result])

        # Add summary to XML structure
        run_result_element = ET.SubElement(
            root_element,
            "GeneratorRunResult",
            trainingData="No training data used",
            version="1.0",
            priority="1",
            trained="F",
            desc="Generated summary",
            link="https://github.com/PV-Credit/ModelRepo"
        )
        topic_result_element = ET.SubElement(run_result_element, "GeneratorTopicResult", topic=topic_id, elapsedTime="5")
        topic_result_element.text = final_summary

    # Save XML
    tree = ET.ElementTree(root_element)
    xml_output_path = os.path.join(output_directory, "summaries.xml")
    tree.write(xml_output_path, encoding='ISO-8859-1', xml_declaration=True)

    # Save detection results to CSV
    with open(results_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['TopicID', 'Detection'])
        writer.writerows(results)

    print("Summaries and detection results saved.")
    logging.info("Summaries and detection results saved.")

# ==============================
# 6. Main Execution
# ==============================

def main():
    try:
        # Initialize tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token = '[PAD]'

        # Initialize policy and value models
        policy_model = GPT2LMHeadModel.from_pretrained('gpt2')
        policy_model.to(device)

        value_model = GPT2LMHeadModel.from_pretrained('gpt2')
        value_model.to(device)

        # Resize token embeddings and set pad_token_id
        policy_model.resize_token_embeddings(len(tokenizer))
        value_model.resize_token_embeddings(len(tokenizer))
        policy_model.config.pad_token_id = tokenizer.pad_token_id
        value_model.config.pad_token_id = tokenizer.pad_token_id

        # Initialize reward model
        reward_tokenizer = RobertaTokenizer.from_pretrained('./Roberta')
        reward_model = RobertaForSequenceClassification.from_pretrained('./Roberta')
        reward_model.to(device)
        reward_model.eval()

        logging.info("Initialized models successfully.")

        # ==============================
        # Load and Prepare Data
        # ==============================

        # Specify your input directory and SGML file paths
        input_directory = "./GenAI24-NIST-pilot-T2T-G-set-2/GenAI24-NIST-pilot-T2T-G-set-2/files"
        output_directory = "./"
        results_file = "./results.csv"
        topics_file = "./GenAI24-NIST-pilot-T2T-G-set-2/GenAI24-NIST-pilot-T2T-G-set-2/GenAI24-NIST-pilot-T2T-G-set-2_topics.sgml"

        # Load articles from SGML and corresponding text files
        articles = load_articles(input_directory, topics_file)
        if not articles:
            logging.error("No articles loaded from the SGML file. Please check the file path and structure.")
            raise ValueError("No articles loaded. Please check the input directory and SGML file.")
        else:
            logging.info(f"Loaded {len(articles)} articles successfully.")

        # Create a Hugging Face Dataset from the articles
        train_dataset = Dataset.from_pandas(pd.DataFrame(articles))

        # Initialize Spacy NLP model
        nlp_model = nlp

        # Start training with A2C
        train_a2c_model(
            policy_model=policy_model,
            value_model=value_model,
            train_dataset=train_dataset,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            tokenizer=tokenizer,
            device=device,
            nlp_model=nlp_model,
        )

        # Process and save summaries, passing the NLP model
        process_and_save_summaries(
            input_directory=input_directory,
            output_directory=output_directory,
            results_file=results_file,
            topics_file=topics_file,
            policy_model=policy_model,
            value_model=value_model,
            tokenizer=tokenizer,
            device=device,
            nlp_model=nlp_model,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer
        )

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
        logging.error(traceback.format_exc())
        print("Execution terminated due to an error.")
    except Exception as e:
        logging.error(f'An error occured during execution: {str(e)}') 

if __name__ == "__main__":
    main()
