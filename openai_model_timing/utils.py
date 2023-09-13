import time
import random
import numpy as np

def send_requests(openai, model_name, input_text, num_requests, debug=False):
    response_times = []  # Collect response times

    if debug:
        print(f"Debug mode: Simulating {num_requests} requests to {model_name}...")
        for _ in range(num_requests):
            sim_response_time = random.uniform(0, 0.1)
            response_times.append(sim_response_time)
    else:
        print(f"Testing {model_name}...")

        for i in range(num_requests):
            start_time = time.time()

            if model_name.startswith('gpt'):
                try:
                    response = openai.ChatCompletion.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": input_text}
                        ],
                        max_tokens=100,
                        temperature=1,
                    )
                except:
                    continue
            else:
                try:
                    response = openai.Completion.create(
                        model=model_name,
                        prompt=input_text,
                        max_tokens=100,  # Adjust as needed
                        temperature=1,
                    )
                except:
                    continue

            end_time = time.time()

            response_time = end_time - start_time
            response_times.append(response_time)

            print(f"Request {i + 1}/{num_requests} to {model_name} took {response_time:.2f} seconds")

    average_time = sum(response_times) / len(response_times)
    std = np.std(response_times)
    print(f"Average response time for {num_requests} requests to {model_name}: {average_time:.2f} seconds")
    return average_time, std, response_times  # Return average time and all response times
