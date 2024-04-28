import torch
import datetime

def evaluate_pytorch_settings():
    settings = {}
    settings['PyTorch Version'] = torch.__version__
    settings['CUDA Availability'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        settings['CUDA Version'] = torch.version.cuda
        settings['CUDA Device Count'] = torch.cuda.device_count()
        settings['CUDA Current Device'] = torch.cuda.current_device()
        settings['CUDA Device Name'] = torch.cuda.get_device_name(torch.cuda.current_device())
        settings['CUDA Device Memory'] = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3) # Convert bytes to GB
    return settings

def log_evaluation_result(result):
    with open("pytorch_settings_log.txt", "a") as log_file:
        log_file.write("Evaluation Date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        for key, value in result.items():
            log_file.write(f"{key}: {value}\n")
        log_file.write("\n")

def main():
    result = evaluate_pytorch_settings()
    print("PyTorch Settings Evaluation Result:")
    for key, value in result.items():
        print(f"{key}: {value}")
    log_evaluation_result(result)

if __name__ == "__main__":
    main()
