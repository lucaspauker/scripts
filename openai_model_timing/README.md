# OpenAI Model Timing Scripts

## Request and log

The simplest script is to just request a set of OpenAI models N times and log the mean response time and standard deviation of the response time.
`python request_and_log.py --api_key <YOUR_KEY> --num_requests 1 --log_filename timing.log`

## Base model latency comparison

`python openai_model_timing.py --api_key <YOUR_KEY> --num_requests 100 --output_bar_plot response_times_bar_graph.png --output_scatter_plot response_times_scatter_plot.png --json_filename response_times.json`

Only generate plots from a `response_times.json` file
`python openai_model_timing.py --api_key <YOUR_KEY> --num_requests 100 --output_bar_plot response_times_bar_graph.png --output_scatter_plot response_times_scatter_plot.png --json_filename response_times.json --plot_only`

For more information, type `python openai_model_timing.py -h`.

## Fine-tune vs. base model latency comparison

`python openai_finetuned_model_timing.py --api_key <YOUR_KEY> --num_requests 100 --output_bar_plot finetuned_response_times_bar_graph.png --json_filename finetuned_response_times.json --model_names ft:gpt-3.5-turbo-0613:personal::XXXXXXXX ft:davinci-002:personal::XXXXXXXX ft:babbage-002:personal::XXXXXXXX`

Only generate plots from a `response_times.json` file
`python openai_finetuned_model_timing.py --api_key <YOUR_KEY> --num_requests 100 --output_bar_plot finetuned_response_times_bar_graph.png --json_filename finetuned_response_times.json --model_names ft:gpt-3.5-turbo-0613:personal::XXXXXXXX ft:davinci-002:personal::XXXXXXXX ft:babbage-002:personal::XXXXXXXX --plot_only`

For more information, type `python openai_finetuned_model_timing.py -h`.

