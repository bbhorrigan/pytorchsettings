import os
import datetime
import logging
import configparser
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ftplib import FTP
import torch
import json

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')  # Assuming configuration details are stored in config.ini

# Initialize logger
logging.basicConfig(filename='pytorch_settings.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to evaluate PyTorch settings stored no your machine
def evaluate_pytorch_settings():
    settings = {}
    try:
        settings['PyTorch Version'] = torch.__version__
        settings['CUDA Availability'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            settings['CUDA Version'] = torch.version.cuda
            settings['CUDA Device Count'] = torch.cuda.device_count()
            settings['CUDA Current Device'] = torch.cuda.current_device()
            settings['CUDA Device Name'] = torch.cuda.get_device_name(torch.cuda.current_device())
            settings['CUDA Device Memory'] = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3) # Convert bytes to GB
    except Exception as e:
        logging.error(f"Error evaluating PyTorch settings: {e}")
    return settings

# Function to log evaluation result
def log_evaluation_result(result):
    try:
        with open("pytorch_settings_log.txt", "a") as log_file:
            log_file.write("Evaluation Date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            for key, value in result.items():
                log_file.write(f"{key}: {value}\n")
            log_file.write("\n")
    except Exception as e:
        logging.error(f"Error logging evaluation result: {e}")

# Function to send email notification
def send_email_notification(subject, body):
    sender_email = config['EMAIL']['SENDER_EMAIL']
    receiver_email = config['EMAIL']['RECEIVER_EMAIL']
    password = config['EMAIL']['PASSWORD']

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(config['EMAIL']['SMTP_SERVER'], int(config['EMAIL']['SMTP_PORT']))
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        logging.info("Email notification sent successfully.")
    except Exception as e:
        logging.error(f"Error sending email notification: {e}")
    finally:
        server.quit()

# Function to upload backup file to FTP server
def upload_to_ftp(file_path):
    ftp_host = config['FTP']['HOST']
    ftp_username = config['FTP']['USERNAME']
    ftp_password = config['FTP']['PASSWORD']

    try:
        with FTP(ftp_host) as ftp:
            ftp.login(user=ftp_username, passwd=ftp_password)
            with open(file_path, 'rb') as f:
                ftp.storbinary(f'STOR {os.path.basename(file_path)}', f)
        logging.info("File uploaded to FTP server successfully.")
    except Exception as e:
        logging.error(f"Error uploading file to FTP server: {e}")

# Function to save model and optimizer states
def save_model_and_optimizer(model, optimizer, epoch, loss, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, path)

# Function to load model and optimizer states
def load_model_and_optimizer(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

# Function to save training configurations
def save_training_config(config_dict, path):
    with open(path, 'w') as config_file:
        json.dump(config_dict, config_file)

# Function to load training configurations
def load_training_config(path):
    with open(path, 'r') as config_file:
        return json.load(config_file)

# Main function
def main():
    try:
        result = evaluate_pytorch_settings()
        logging.info("PyTorch Settings Evaluation Result:")
        for key, value in result.items():
            logging.info(f"{key}: {value}")
        log_evaluation_result(result)

        # Save model and optimizer states
        model = ... # Your model
        optimizer = ... # Your optimizer
        epoch = ... # Your current epoch
        loss = ... # Your current loss
        save_model_and_optimizer(model, optimizer, epoch, loss, 'path/to/checkpoint.pth')

        # Save training configurations
        training_config = {
            'learning_rate': ... , # Your learning rate
            'batch_size': ... , # Your batch size
            'num_epochs': ... , # Your number of epochs
        }
        save_training_config(training_config, 'path/to/config.json')

        # Send email notification
        subject = "PyTorch Settings Backup Complete"
        body = "PyTorch settings have been successfully backed up."
        send_email_notification(subject, body)

        # Upload backup file to FTP server
        file_path = "pytorch_settings_log.txt"
        if os.path.exists(file_path):
            upload_to_ftp(file_path)
        else:
            logging.error("Backup file not found.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
