
import re

ABSOLUTIST_WORDS = {
    'always', 'never', 'nothing', 'everything', 'completely', 'totally',
    'absolutely', 'entirely', 'constantly', 'forever', 'impossible',
    'definitely', 'certainly', 'wholly', 'utterly', 'all', 'none',
    'every', 'no one', 'everyone', 'nobody', 'everybody'
}

NEGATIVE_WORDS = {
    'sad', 'depressed', 'anxious', 'worried', 'afraid', 'scared', 'angry',
    'frustrated', 'hopeless', 'worthless', 'useless', 'terrible', 'awful',
    'horrible', 'miserable', 'lonely', 'alone', 'empty', 'numb', 'exhausted',
    'tired', 'overwhelmed', 'stressed', 'panic', 'fear', 'hate', 'crying',
    'tears', 'pain', 'hurt', 'suffering', 'struggling', 'failed', 'failure',
    'broken', 'damaged', 'lost', 'confused', 'helpless', 'desperate'
}

CRISIS_WORDS = {
    'suicide', 'suicidal', 'kill myself', 'end it', 'give up', 'cant go on',
    "can't go on", 'no point', 'want to die', 'better off dead', 'end my life',
    'self harm', 'self-harm', 'cutting', 'overdose', 'pills'
}

DISTORTION_WORDS = {
    'should', 'must', 'have to', 'need to', 'cant', "can't", 'wont', "won't",
    'ruined', 'disaster', 'catastrophe', 'worst', 'never going to'
}

FIRST_PERSON = {'i', 'me', 'my', 'myself', 'mine', "i'm", "i've", "i'll", "i'd"}

def calculate_linguistic_risk(text):
    if not text or not isinstance(text, str):
        return {'risk_score': 0, 'components': {}, 'detected_words': {}, 'word_count': 0}
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    word_count = len(words) if words else 1
    
    absolutist_count = sum(1 for w in words if w in ABSOLUTIST_WORDS)
    negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    first_person_count = sum(1 for w in words if w in FIRST_PERSON)
    distortion_count = sum(1 for w in words if w in DISTORTION_WORDS)
    crisis_count = sum(1 for phrase in CRISIS_WORDS if phrase in text_lower)
    
    normalize = 100 / word_count
    
    components = {
        'absolutist': min(absolutist_count * normalize * 3, 20),
        'negative_emotion': min(negative_count * normalize * 2, 25),
        'first_person': min(first_person_count * normalize * 1, 15),
        'cognitive_distortion': min(distortion_count * normalize * 3, 20),
        'crisis_language': min(crisis_count * 10, 20)
    }
    
    risk_score = sum(components.values())
    
    detected = {
        'absolutist': [w for w in words if w in ABSOLUTIST_WORDS],
        'negative': [w for w in words if w in NEGATIVE_WORDS],
        'crisis': [p for p in CRISIS_WORDS if p in text_lower]
    }
    
    return {
        'risk_score': round(risk_score, 1),
        'components': components,
        'detected_words': detected,
        'word_count': word_count
    }

def get_risk_level(score):
    if score < 15:
        return 'Low', '🟢'
    elif score < 30:
        return 'Moderate', '🟡'
    elif score < 50:
        return 'Elevated', '🟠'
    else:
        return 'High', '🔴'
