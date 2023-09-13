import openai
import datetime
import argparse

from utils import send_requests

def setup_openai(api_key):
    openai.api_key = api_key
    openai.api_requestor.TIMEOUT_SECS = 10

def main():
    parser = argparse.ArgumentParser(description="Test OpenAI models with multiple requests.")
    parser.add_argument("--api_key", required=True, help="Your OpenAI API key")
    parser.add_argument("--num_requests", type=int, required=True, help="Number of requests to send")
    parser.add_argument("--log_filename", default="response_times.json", help="Output filename for the JSON data")
    args = parser.parse_args()

    setup_openai(args.api_key)

    models = [
        'gpt-4',
        'gpt-3.5-turbo',
    ]

    input_text = "Once upon a time, in a land far, far away..."

    for model_name in models:
        print("\n--------------------------")
        print(f"Testing model: {model_name}")
        average_time, std, response_times = send_requests(openai, model_name, input_text, args.num_requests, False)
        with open(args.log_filename, 'a') as f:
            f.write(f"{str(datetime.datetime.now())},{model_name},{average_time},{std}\n")

    print("Success")

if __name__ == "__main__":
    main()
