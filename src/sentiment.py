import os

# Dynamic import of VADER sentiment analyzer
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    analyzer = SentimentIntensityAnalyzer()
    print("VADER Sentiment Analyzer loaded successfully.")
except ImportError:
    VADER_AVAILABLE = False
    print("vaderSentiment not installed. Using basic keyword sentiment analyzer.")

# Optional Hugging Face transformers sentiment import
TRANSFORMERS_AVAILABLE = False
hf_pipeline = None

def load_huggingface_sentiment():
    """Tries to load Hugging Face DistilBERT sentiment pipeline. Slow on CPU, use with caution."""
    global TRANSFORMERS_AVAILABLE, hf_pipeline
    try:
        from transformers import pipeline
        # Load pre-trained DistilBERT pipeline for sentiment analysis
        hf_pipeline = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1 # Use CPU by default
        )
        TRANSFORMERS_AVAILABLE = True
        print("Hugging Face DistilBERT Sentiment Analysis Pipeline loaded successfully.")
    except Exception as e:
        TRANSFORMERS_AVAILABLE = False
        print(f"Failed to load HuggingFace Pipeline: {e}. Falling back to VADER.")

def get_lexicon_sentiment(text):
    """Simple dictionary-based sentiment analyzer for zero-dependency fallback."""
    text_lower = str(text).lower()
    
    pos_words = {"great", "awesome", "excellent", "amazing", "love", "good", "beautiful", "wonderful", "masterpiece", "perfect", "fun"}
    neg_words = {"bad", "terrible", "worst", "waste", "boring", "slow", "disappointing", "hate", "awful", "horrible", "fail", "snooze"}
    
    words = text_lower.split()
    pos_count = sum(1 for w in words if w in pos_words)
    neg_count = sum(1 for w in words if w in neg_words)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total

def analyze_sentiment(text, method="vader"):
    """
    Analyzes sentiment of the text.
    Returns:
        Score between -1.0 (extremely negative) and 1.0 (extremely positive).
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return 0.0
        
    if method == "huggingface":
        if hf_pipeline is None:
            load_huggingface_sentiment()
            
        if TRANSFORMERS_AVAILABLE and hf_pipeline is not None:
            try:
                # DistilBERT returns: [{'label': 'POSITIVE', 'score': 0.99}]
                # We limit text to 512 tokens to avoid DistilBERT crash
                truncated_text = text[:1000] 
                res = hf_pipeline(truncated_text)[0]
                label = res['label']
                conf = res['score']
                
                # Transform to [-1.0, 1.0] scale
                if label == 'POSITIVE':
                    return conf
                else:
                    return -conf
            except Exception as e:
                print(f"HF Sentiment analysis failed, using VADER fallback. Error: {e}")
                # Fall through to VADER
        else:
            method = "vader"
            
    if method == "vader" and VADER_AVAILABLE:
        scores = analyzer.polarity_scores(text)
        return scores['compound'] # Compound score is normalized between -1 and +1
        
    # Standard lexicon fallback
    return get_lexicon_sentiment(text)

if __name__ == "__main__":
    test_text_pos = "I absolutely loved this film! The acting was fantastic."
    test_text_neg = "It was a total waste of time, incredibly boring and slow."
    
    print(f"Test Positive: {analyze_sentiment(test_text_pos)}")
    print(f"Test Negative: {analyze_sentiment(test_text_neg)}")
