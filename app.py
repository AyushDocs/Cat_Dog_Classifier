import gradio as gr
from src.config import MODEL_MAP
from src.utils import load_model_weights, predict

loaded_models = {}


def load_selected_model(model_name):
    if model_name not in loaded_models:
        loaded_models[model_name] = load_model_weights(model_name)
    return loaded_models[model_name]


def classify(image, model_name):
    if image is None:
        return "Please upload an image.", None
    try:
        model = load_selected_model(model_name)
        result = predict(model, image)
        probs = result["probabilities"]
        bar_chart = {k: float(v) for k, v in probs.items()}
        label = f"{result['class'].upper()} ({result['confidence']:.1%})"
        return label, bar_chart
    except Exception as e:
        return f"Error: {str(e)}", None


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
        gr.BarPlot(x="key", y="value", label="Confidence", x_label="Class", y_label="Probability"),
    ],
    title="Cat vs Dog Classifier",
    description="Classify images of cats and dogs using different trained models.",
    examples=[
        ["research/sample_cat.jpg", "Transfer Learning (Layer4 + FC)"],
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
