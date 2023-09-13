import openai
import argparse
import time
import random
import matplotlib.pyplot as plt
import json
import numpy as np

from utils import send_requests

def setup_openai(api_key):
    openai.api_key = api_key
    openai.api_requestor.TIMEOUT_SECS = 10

def create_and_save_bar_plot(json_filename, output_filename):
    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)
        num_requests = data["num_requests"]
        input_prompt = data["input_prompt"]
        model_data = data["model_data"]

    model_names = list(model_data.keys())
    avg_times = [model_data[x]["mean"] for x in model_names]
    stds = [model_data[x]["std"] for x in model_names]
    plt.figure(figsize=(10, 6))

    r, g, b = "#e60049", "#0bb4ff", "#50e991"
    colors = [g,b,g,b,g,b]
    plt.bar(model_names, avg_times, yerr=stds, capsize=5, color=colors)
    plt.xlabel("Model")
    plt.ylabel("Average Response Time (seconds)")
    plt.title(f"Average Response Time for OpenAI Models Over {num_requests} Requests")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--")

    legend_labels = [
        plt.Line2D([0], [0], color=g, lw=4, label='Base'),
        plt.Line2D([0], [0], color=b, lw=4, label='Fine-tuned')
    ]
    plt.legend(handles=legend_labels)

    plt.savefig(output_filename)

def main():
    parser = argparse.ArgumentParser(description="Compare fine-tuned models to base models with OpenAI.")
    parser.add_argument("--api_key", required=True, help="Your OpenAI API key")
    parser.add_argument("--num_requests", type=int, required=True, help="Number of requests to send")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (simulated requests)")
    parser.add_argument("--model_names", nargs=3, required=True, help="List of model names to compare")
    parser.add_argument("--output_bar_plot", default="response_times.png", help="Output filename for the bar plot")
    parser.add_argument("--json_filename", default="response_times.json", help="Output filename for the JSON data")
    parser.add_argument("--plot_only", action="store_true", help="Plot only, do not send requests")
    args = parser.parse_args()

    if not args.plot_only:
        setup_openai(args.api_key)

        input_text = "—Yo soy ardiente, yo soy morena, yo soy el símbolo de la pasión, de ansia de goces mi alma está llena.  ¿A mí me buscas?  —No es a ti, no.  —Mi frente es pálida, mis trenzas de oro: puedo brindarte dichas sin fin, yo de ternuras guardo un tesoro.  ¿A mí me llamas?  —No, no es a ti.  —Yo soy un sueño, un imposible, vano fantasma de niebla y luz; soy incorpórea, soy intangible: no puedo amarte.  —¡Oh ven, ven tú!  "

        data = {}  # Collect average times for each model

        base_models = [
            'gpt-3.5-turbo',
            'davinci-002',
            'babbage-002',
        ]

        for i in range(len(args.model_names)):
            model_name = base_models[i]
            print("\n--------------------------")
            print(f"Testing model: {model_name}")
            average_time, std, response_times = send_requests(openai, model_name, input_text, args.num_requests, args.debug)
            data[model_name] = {
                    "mean": average_time,
                    "std": std,
                }

            model_name = args.model_names[i]
            print("\n--------------------------")
            print(f"Testing model: {model_name}")
            average_time, std, response_times = send_requests(model_name, input_text, args.num_requests, args.debug)
            data[base_models[i] + " fine-tuned"] = {
                    "mean": average_time,
                    "std": std,
                }

        # Create and save the JSON data
        json_data = {
            "num_requests": args.num_requests,
            "input_prompt": input_text,
            "model_data": data,
        }
        with open(args.json_filename, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)

    # Create and save the bar plot
    create_and_save_bar_plot(args.json_filename, args.output_bar_plot)

if __name__ == "__main__":
    main()

