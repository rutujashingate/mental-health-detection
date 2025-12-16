
import re
from collections import Counter

def highlight_risk_words(text):
    """Identify and highlight risk-related words in text."""
    if not text or not isinstance(text, str):
        return {'highlighted_words': [], 'categories': {}}
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    categories = {
        'negative_emotion': {'sad', 'depressed', 'anxious', 'worried', 'afraid', 'scared', 
                            'hopeless', 'worthless', 'miserable', 'lonely', 'empty', 'exhausted',
                            'overwhelmed', 'stressed', 'panic', 'hate', 'pain', 'hurt', 'helpless'},
        'absolutist': {'always', 'never', 'nothing', 'everything', 'completely', 'totally',
                      'impossible', 'forever', 'constantly'},
        'crisis': {'suicide', 'suicidal', 'die', 'death', 'end', 'give up'},
        'cognitive': {'should', 'must', 'cant', "can't", 'ruined', 'failed', 'failure', 'worst'}
    }
    
    detected = {cat: [] for cat in categories}
    
    for word in words:
        for category, word_set in categories.items():
            if word in word_set:
                detected[category].append(word)
    
    all_detected = []
    for cat, words_list in detected.items():
        for w in words_list:
            all_detected.append({'word': w, 'category': cat})
    
    return {
        'highlighted_words': all_detected,
        'categories': {k: list(set(v)) for k, v in detected.items() if v}
    }

# Condition-specific keywords for prediction explanation
CONDITION_KEYWORDS = {
    'depression': ['hopeless', 'worthless', 'empty', 'sad', 'crying', 'alone', 'tired', 'exhausted', 'numb'],
    'ADHD': ['focus', 'distracted', 'hyperactive', 'impulsive', 'attention', 'concentrate', 'forgot', 'late'],
    'OCD': ['obsessive', 'compulsive', 'intrusive', 'ritual', 'contamination', 'checking', 'counting', 'symmetry'],
    'ptsd': ['trauma', 'flashback', 'nightmare', 'trigger', 'hypervigilant', 'startle', 'avoid', 'numb'],
    'aspergers': ['social', 'sensory', 'routine', 'literal', 'eye contact', 'special interest', 'masking']
}

def explain_prediction(text, predicted_class, confidence):
    """Generate explanation for why a prediction was made."""
    text_lower = text.lower()
    
    # Find matching keywords
    keywords_found = []
    for keyword in CONDITION_KEYWORDS.get(predicted_class, []):
        if keyword in text_lower:
            keywords_found.append(keyword)
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'keywords_found': keywords_found,
        'explanation': f"Detected {len(keywords_found)} keywords associated with {predicted_class}"
    }
