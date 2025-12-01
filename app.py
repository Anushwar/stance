"""
Stance Detection Web Application
Python-based web UI for showcasing stance detection models

Based on the UI implementation plan from the project presentation:
- Input: Tweet text and target selection
- Output: Predicted stance with confidence scores
- Model selection: Choose between different trained models
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Check for gradio
try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio is not installed.")
    print("Please install it with: pip install gradio")
    print("\nOr install all dependencies: pip install -r requirements.txt")
    sys.exit(1)

from src.models.inference import StanceDetector, get_available_models

# Targets from SemEval-2016 dataset
TARGETS = [
    "Atheism",
    "Climate Change is a Real Concern",
    "Feminist Movement",
    "Hillary Clinton",
    "Legalization of Abortion"
]

# Model mapping
MODEL_MAP = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "K-Nearest Neighbors": "knn",
    "BERT-base": "bert",
    "BERTweet": "bertweet",
    "TwHIN-BERT": "twhin_bert"
}

# Global detector cache
detectors = {}

def get_detector(model_name):
    """Get or create detector for the specified model"""
    if model_name not in detectors:
        model_key = MODEL_MAP.get(model_name, "logistic_regression")
        try:
            detectors[model_name] = StanceDetector(model_key)
        except Exception as e:
            return None, str(e)
    return detectors[model_name], None

def predict_stance(tweet, target, model_name):
    """
    Predict stance for a tweet-target pair

    Args:
        tweet: Tweet text
        target: Target entity
        model_name: Selected model name

    Returns:
        tuple: (stance_text, confidence_html)
    """
    if not tweet or not tweet.strip():
        return "Please enter a tweet", ""

    if not target:
        return "Please select a target", ""

    # Get detector
    detector, error = get_detector(model_name)
    if error:
        return f"Model not available: {error}", ""

    # Predict
    try:
        result = detector.predict(tweet, target)

        # Format stance output
        stance = result['stance']
        confidence = result['confidence']

        # Create color-coded output
        stance_colors = {
            'AGAINST': '#ef4444',  # red
            'FAVOR': '#10b981',    # green
            'NONE': '#6b7280'      # gray
        }

        stance_text = f"""
        <div style='text-align: center; padding: 20px;'>
            <h2 style='color: {stance_colors.get(stance, "#000")}; margin: 0;'>{stance}</h2>
            <p style='font-size: 18px; color: #666; margin-top: 10px;'>
                Confidence: {confidence:.1%}
            </p>
        </div>
        """

        # Format probability bars
        probabilities = result['probabilities']
        prob_html = "<div style='padding: 10px;'>"

        for stance_label in ['FAVOR', 'AGAINST', 'NONE']:
            prob = probabilities.get(stance_label, 0)
            color = stance_colors.get(stance_label, "#666")

            prob_html += f"""
            <div style='margin-bottom: 15px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span style='font-weight: bold;'>{stance_label}</span>
                    <span>{prob:.1%}</span>
                </div>
                <div style='width: 100%; background-color: #e5e7eb; border-radius: 10px; overflow: hidden;'>
                    <div style='width: {prob*100}%; background-color: {color}; height: 25px; transition: width 0.3s;'></div>
                </div>
            </div>
            """

        prob_html += "</div>"

        return stance_text, prob_html

    except Exception as e:
        return f"Error during prediction: {str(e)}", ""

# Example tweets for demonstration
examples = [
    [
        "I will fight for the unborn!",
        "Legalization of Abortion",
        "Logistic Regression"
    ],
    [
        "Climate change is the greatest threat facing humanity. We must act now!",
        "Climate Change is a Real Concern",
        "Random Forest"
    ],
    [
        "Everyone is entitled to their own beliefs. #Freedom",
        "Atheism",
        "K-Nearest Neighbors"
    ],
    [
        "Women deserve equal rights and opportunities in all aspects of life.",
        "Feminist Movement",
        "Logistic Regression"
    ],
    [
        "She has the experience and leadership we need.",
        "Hillary Clinton",
        "Random Forest"
    ]
]

# Get available models
available_models = get_available_models()

# Create Gradio interface
with gr.Blocks(title="Stance Detection on Social Media") as demo:

    gr.Markdown(
        """
        # Stance Detection on Social Media

        ### Detect stance (FAVOR, AGAINST, NONE) in tweets using state-of-the-art models

        This application uses transformer-based models fine-tuned on the SemEval-2016 stance detection dataset.
        Enter a tweet and select a target to see the predicted stance.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")

            tweet_input = gr.Textbox(
                label="Tweet Text",
                placeholder="Enter your tweet here...",
                lines=4,
                max_lines=6
            )

            target_input = gr.Dropdown(
                choices=TARGETS,
                label="Target",
                value=TARGETS[0]
            )

            model_input = gr.Dropdown(
                choices=available_models,
                label="Model",
                value=available_models[0] if available_models else None
            )

            predict_btn = gr.Button("Detect Stance", variant="primary", size="lg")

            gr.Markdown(
                """
                **Available Models:**
                - **Traditional ML**: Logistic Regression with TF-IDF features
                - **BERT-base**: General-purpose transformer (110M params)
                - **BERTweet**: Twitter-specific BERT (110M params, 850M tweets)
                - **TwHIN-BERT**: State-of-the-art Twitter BERT (110M params, 7B tweets)
                """
            )

        with gr.Column(scale=1):
            gr.Markdown("### Output")

            stance_output = gr.HTML(label="Predicted Stance")

            gr.Markdown("### Confidence Scores")
            prob_output = gr.HTML(label="Probabilities")

    # Connect the predict button
    predict_btn.click(
        fn=predict_stance,
        inputs=[tweet_input, target_input, model_input],
        outputs=[stance_output, prob_output]
    )

    # Add examples
    gr.Markdown("### Example Tweets")
    gr.Examples(
        examples=examples,
        inputs=[tweet_input, target_input, model_input],
        outputs=[stance_output, prob_output],
        fn=predict_stance,
        cache_examples=False
    )

    gr.Markdown(
        """
        ---

        ### About This Project

        This stance detection system was developed as part of CSE 573 - Semantic Web Mining.
        The models are trained on the SemEval-2016 Task 6 dataset containing 4,063 annotated tweets
        across 5 controversial targets.

        **Key Insight:** Stance ≠ Sentiment
        - 30.9% of AGAINST tweets have POSITIVE sentiment
        - 56.4% of FAVOR tweets have NEGATIVE sentiment

        This is why specialized stance detection models are necessary!

        **Project Repository:** [GitHub](https://github.com/Anushwar/stance)
        """
    )

# Launch the app
if __name__ == "__main__":
    print("="*80)
    print("STANCE DETECTION WEB APPLICATION")
    print("="*80)
    print(f"\nAvailable models: {available_models}")
    print("\nLaunching Gradio interface...")
    print("="*80)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
