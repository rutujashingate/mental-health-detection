import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import re
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(
    page_title="Mental Health Risk Analyzer",
    # page_icon="🧠",
    layout="wide"
)

# ============== RISK CALCULATOR ==============
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
    'self harm', 'self-harm', 'cutting', 'overdose'
}

DISTORTION_WORDS = {
    'should', 'must', 'have to', 'need to', 'cant', "can't", 'wont', "won't",
    'ruined', 'disaster', 'catastrophe', 'worst', 'never going to'
}

FIRST_PERSON = {'i', 'me', 'my', 'myself', 'mine', "i'm", "i've", "i'll", "i'd"}

def calculate_linguistic_risk(text):
    if not text or not isinstance(text, str):
        return {'risk_score': 0, 'components': {}, 'detected_words': {}}
    
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
        'Absolutist Language': min(absolutist_count * normalize * 3, 20),
        'Negative Emotion': min(negative_count * normalize * 2, 25),
        'Self-Focus': min(first_person_count * normalize * 1, 15),
        'Cognitive Distortion': min(distortion_count * normalize * 3, 20),
        'Crisis Language': min(crisis_count * 10, 20)
    }
    
    risk_score = sum(components.values())
    
    detected = {
        'absolutist': [w for w in words if w in ABSOLUTIST_WORDS],
        'negative': [w for w in words if w in NEGATIVE_WORDS],
        'crisis': [p for p in CRISIS_WORDS if p in text_lower],
        'distortion': [w for w in words if w in DISTORTION_WORDS]
    }
    
    return {
        'risk_score': round(risk_score, 1),
        'components': components,
        'detected_words': detected,
        'word_count': word_count
    }

def get_risk_level(score):
    if score < 15:
        return 'Low', '#28a745'
    elif score < 30:
        return 'Moderate', '#ffc107'
    elif score < 50:
        return 'Elevated', '#fd7e14'
    else:
        return 'High', '#dc3545'

# ============== BILSTM MODEL ==============
class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, num_classes=5, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, seq_lens=None):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out).squeeze(-1)
        
        if seq_lens is not None:
            mask = torch.zeros_like(attn_weights)
            for i, length in enumerate(seq_lens):
                mask[i, length:] = float('-inf')
            attn_weights = attn_weights + mask
        
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return self.classifier(context), attn_weights

# ============== LOAD MODELS ==============
@st.cache_resource
def load_models():
    # Load sentence transformer
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load BiLSTM model
    device = torch.device('cpu')
    model = BiLSTMAttention(input_dim=384, hidden_dim=128, num_classes=5)
    
    try:
        model.load_state_dict(torch.load('results/bilstm_model.pth', map_location=device))
        model.eval()
        model_loaded = True
    except:
        model_loaded = False
        model = None
    
    class_names = ['ADHD', 'OCD', 'aspergers', 'depression', 'ptsd']
    
    return encoder, model, model_loaded, class_names, device

# ============== MAIN APP ==============
def main():
    # Header
    st.title("Mental Health Risk Analyzer")
    st.markdown("""
    This tool analyzes text for linguistic patterns associated with mental health conditions.
    It provides **risk scores** based on research-backed linguistic markers and **condition predictions** 
    using a BiLSTM deep learning model.
    """)
    
    st.warning("**Disclaimer**: This tool is for educational/research purposes only. It is NOT a clinical diagnostic tool and should never replace professional mental health assessment.")
    
    # Load models
    encoder, model, model_loaded, class_names, device = load_models()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    **How it works:**
    1. **Linguistic Risk Score** analyzes text for:
       - Absolutist language (always, never)
       - Negative emotion words
       - Self-focused language
       - Cognitive distortions
       - Crisis language
    
    2. **Condition Prediction** uses a BiLSTM model trained on Reddit mental health posts.
    
    **Research basis:**
    - Pennebaker (2011): Language and self-disclosure
    - Al-Mosaiwi & Johnstone (2018): Absolutist thinking in mental health
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Performance:**")
    st.sidebar.markdown("""
    - Overall Accuracy: 93.8%
    - Best: OCD (97.9%)
    - Challenging: ADHD (88.4%)
    """)
    
    # Main input
    st.header("Enter Text to Analyze")
    
    # Example texts
    example_texts = {
        "Select an example...": "",
        "Low Risk": "I had a pretty good day today. Went for a walk and caught up with a friend. Feeling okay about things.",
        "Moderate Risk": "I've been feeling a bit anxious lately about work. Some days are harder than others, but I'm managing.",
        "Elevated Risk": "I always feel like I'm failing at everything. Nothing I do is ever good enough and I'm so tired of trying.",
        "High Risk": "I can't do this anymore. Everything is hopeless and I feel completely worthless. Nothing will ever get better."
    }
    
    selected_example = st.selectbox("Or choose an example:", list(example_texts.keys()))
    
    if selected_example != "Select an example...":
        default_text = example_texts[selected_example]
    else:
        default_text = ""
    
    user_text = st.text_area(
        "Enter text (social media post, journal entry, etc.):",
        value=default_text,
        height=150,
        placeholder="Type or paste text here..."
    )
    
    analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_button and user_text.strip():
        st.markdown("---")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        # ============== LINGUISTIC RISK ANALYSIS ==============
        with col1:
            st.header("Linguistic Risk Analysis")
            
            risk_result = calculate_linguistic_risk(user_text)
            risk_score = risk_result['risk_score']
            level, color = get_risk_level(risk_score)

            # Risk score display
            st.markdown(f"""
            <div style="background-color: {color}22; border-left: 5px solid {color}; padding: 20px; border-radius: 5px; margin-bottom: 20px;">
                <h2 style="margin: 0; color: {color};">Risk Level: {level}</h2>
                <h1 style="margin: 10px 0; font-size: 48px;">{risk_score}/100</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Component breakdown
            st.subheader("Score Components")
            for component, score in risk_result['components'].items():
                percentage = score / 25 * 100  # Normalize for display
                st.markdown(f"**{component}**: {score:.1f}")
                st.progress(min(score / 25, 1.0))
            
            # Detected words
            st.subheader("Detected Risk Words")
            detected = risk_result['detected_words']
            
            if detected['negative']:
                st.markdown(f"**Negative Emotion:** {', '.join(set(detected['negative']))}")
            if detected['absolutist']:
                st.markdown(f"**Absolutist:** {', '.join(set(detected['absolutist']))}")
            if detected['distortion']:
                st.markdown(f"**Cognitive Distortion:** {', '.join(set(detected['distortion']))}")
            if detected['crisis']:
                st.error(f"**Crisis Language Detected:** {', '.join(set(detected['crisis']))}")
            
            if not any(detected.values()):
                st.success("No significant risk words detected.")
        
        # ============== CONDITION PREDICTION ==============
        with col2:
            st.header("Condition Prediction")
            
            if model_loaded and model is not None:
                # Encode text
                embedding = encoder.encode([user_text])
                embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0)  # (1, 1, 384)
                
                # Predict
                with torch.no_grad():
                    logits, attention = model(embedding_tensor, seq_lens=torch.tensor([1]))
                    probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
                
                # Sort by probability
                sorted_indices = np.argsort(probabilities)[::-1]
                
                # Display predictions
                st.markdown("**Confidence Scores:**")
                
                condition_colors = {
                    'depression': '#E53935',
                    'ptsd': '#FB8C00',
                    'aspergers': '#7CB342',
                    'ADHD': '#1E88E5',
                    'OCD': '#8E24AA'
                }
                
                for idx in sorted_indices:
                    condition = class_names[idx]
                    prob = probabilities[idx]
                    color = condition_colors.get(condition, '#666666')
                    
                    st.markdown(f"**{condition.upper()}**: {prob*100:.1f}%")
                    st.progress(float(prob))
                
                # Top prediction
                top_condition = class_names[sorted_indices[0]]
                top_prob = probabilities[sorted_indices[0]]
                
                st.markdown("---")
                st.markdown(f"""
                <div style="background-color: #666; padding: 15px; border-radius: 5px;">
                    <h3 style="margin: 0;">Primary Prediction: {top_condition.upper()}</h3>
                    <p style="margin: 5px 0;">Confidence: {top_prob*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Condition-specific insights
                st.subheader("Condition Insights")
                
                insights = {
                    'depression': "Depression posts often contain hopelessness, worthlessness, and self-critical language.",
                    'ptsd': "PTSD posts frequently mention trauma, triggers, flashbacks, and hypervigilance.",
                    'OCD': "OCD posts typically discuss intrusive thoughts, compulsions, and ritualistic behaviors.",
                    'ADHD': "ADHD posts often mention focus issues, forgetfulness, and medication experiences.",
                    'aspergers': "Aspergers posts frequently discuss social challenges, sensory issues, and special interests."
                }
                
                st.info(insights.get(top_condition, ""))
                
            else:
                st.error("Model not loaded. Please ensure 'results/bilstm_model.pth' exists.")
                st.markdown("Run the training notebook first to generate the model file.")
        
        # ============== INTERPRETATION GUIDE ==============
        st.markdown("---")
        st.header("How to Interpret Results")
        
        interpret_col1, interpret_col2 = st.columns(2)
        
        with interpret_col1:
            st.markdown("""
            **Risk Levels:**
            - **Low (0-15)**: Minimal distress indicators
            - **Moderate (15-30)**: Some distress signals present
            - **Elevated (30-50)**: Notable distress patterns
            - **High (50+)**: Significant distress indicators
            """)
        
        with interpret_col2:
            st.markdown("""
            **Limitations:**
            - Based on linguistic patterns, not clinical diagnosis
            - Single text analysis may not reflect overall state
            - Context and intent are not fully captured
            - Should never replace professional assessment
            """)
    
    elif analyze_button:
        st.warning("Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with Streamlit | Research Project for ML Portfolio</p>
        <p>Model trained on Reddit mental health communities data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()




