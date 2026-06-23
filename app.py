import gradio as gr
from src.config import MODEL_MAP
from src.utils import load_model_weights, predict

loaded_models = {}


def classify(image, model_name):
    if image is None:
        return "Please upload an image.", {}
    try:
        model = load_model_weights(model_name)
        result = predict(model, image)
        label = f"{result['class'].upper()} ({result['confidence']:.1%})"
        probs = {k: float(v) for k, v in result["probabilities"].items()}
        return label, probs
    except Exception as e:
        return f"Error: {str(e)}", {}


demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(
            choices=list(MODEL_MAP.keys()),
            value="Transfer Learning (Layer4 + FC)",
            label="Model",
        ),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Confidence"),
    ],
    title="Cat vs Dog Classifier",
    description="Classify images of cats and dogs using different trained models.",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
