import openai
import argparse
import time
import random
import json
import matplotlib.pyplot as plt
import numpy as np

from utils import send_requests
from sklearn.linear_model import LinearRegression

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
    colors = []
    labels = []

    r, g, b = "#e60049", "#0bb4ff", "#50e991"

    for model_name in model_names:
        if model_name.startswith('gpt'):
            colors.append(r)
            labels.append('Chat')
        elif model_name.startswith('text'):
            colors.append(g)
            labels.append('InstructGPT')
        else:
            colors.append(b)
            labels.append('Base')

    avg_times = [model_data[x]["mean"] for x in model_names]
    stds = [model_data[x]["std"] for x in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, avg_times, yerr=stds, capsize=5, color=colors)
    plt.xlabel("Model")
    plt.ylabel("Average Response Time (seconds)")
    plt.title(f"Average Response Time for OpenAI Models over {num_requests} Requests")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--")

    legend_labels = [
        plt.Line2D([0], [0], color=r, lw=4, label='Chat'),
        plt.Line2D([0], [0], color=g, lw=4, label='Instruct'),
        plt.Line2D([0], [0], color=b, lw=4, label='Base')
    ]
    plt.legend(handles=legend_labels)

    plt.savefig(output_filename)

def create_and_save_scatter_plot(json_filename, output_filename):
    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)
        model_data = data["model_data"]

    model_names = list(model_data.keys())
    avg_times = [model_data[x]["mean"] for x in model_names]
    stds = [model_data[x]["std"] for x in model_names]
    # https://learn.microsoft.com/en-us/semantic-kernel/prompt-engineering/llm-models
    num_parameters = {
        'gpt-4': 1.7e12,
        'gpt-4-0613': 1.7e12,
        'gpt-4-32k': 1.7e12,
        'gpt-4-32k-0613': 1.7e12,
        'gpt-3.5-turbo': 175e9,
        'gpt-3.5-turbo-0613': 175e9,
        'gpt-3.5-turbo-16k': 175e9,
        'gpt-3.5-turbo-16k-0613': 175e9,
        'babbage-002': 3e9,
        'davinci-002': 3e9,
        'text-ada-001': 350e6,
        'text-davinci-003': 175e9,
        'text-davinci-002': 175e9,
        'text-davinci-001': 175e9,
        'text-curie-001': 13e9,
        'text-babbage-001': 3e9,
        'text-ada-001': 350e6,
        'davinci': 175e9,
        'curie': 13e9,
        'babbage': 3e9,
        'ada': 350e6,
    }
    num_params = [num_parameters[model_name] for model_name in model_names]

    # Perform linear regression
    # X = np.array(num_params).reshape(-1, 1)  # Reshape to a 2D array
    # y = np.array(avg_times)

    # model = LinearRegression()
    # model.fit(X, y)

    # X_fit = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    # y_fit = model.predict(X_fit)

    plt.figure(figsize=(10, 6))
    plt.scatter(num_params, avg_times, label="Data")
    # plt.plot(X_fit, y_fit, color='red', label=f"Linear Fit (R^2={model.score(X, y):.2f})")

    plt.xlabel("Number of Parameters")
    plt.ylabel("Average Response Time (seconds)")
    plt.title(f"Number of Parameters vs. Response Time")
    plt.xscale("log")
    plt.tight_layout()
    plt.grid()
    plt.savefig(output_filename)

def main():
    parser = argparse.ArgumentParser(description="Test OpenAI models with multiple requests.")
    parser.add_argument("--api_key", required=True, help="Your OpenAI API key")
    parser.add_argument("--num_requests", type=int, required=True, help="Number of requests to send")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (simulated requests)")
    parser.add_argument("--output_bar_plot", default="response_times.png", help="Output filename for the bar plot")
    parser.add_argument("--output_scatter_plot", default="scatter_plot.png", help="Output filename for the scatter plot")
    parser.add_argument("--json_filename", default="response_times.json", help="Output filename for the JSON data")
    parser.add_argument("--plot_only", action="store_true", help="Plot only, do not send requests")
    args = parser.parse_args()

    if not args.plot_only:
        setup_openai(args.api_key)

        models = [
            'gpt-4',
            'gpt-4-0613',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-0613',
            'gpt-3.5-turbo-16k',
            'gpt-3.5-turbo-16k-0613',
            'text-davinci-003',
            'text-davinci-002',
            'text-davinci-001',
            'text-curie-001',
            'text-babbage-001',
            'text-ada-001',
            'davinci-002',
            'babbage-002',
            'davinci',
            'curie',
            'babbage',
            'ada',
        ]

        input_text = "Once upon a time, in a land far, far away..."

        data = {}

        for model_name in models:
            print("\n--------------------------")
            print(f"Testing model: {model_name}")
            average_time, std, response_times = send_requests(openai, model_name, input_text, args.num_requests, args.debug)
            data[model_name] = {
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
            json.dump(json_data, json_file)

    # Create and save plots
    create_and_save_bar_plot(args.json_filename, args.output_bar_plot)
    create_and_save_scatter_plot(args.json_filename, args.output_scatter_plot)
    print("Successfully created plots")

if __name__ == "__main__":
    main()

