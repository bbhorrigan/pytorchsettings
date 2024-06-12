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
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
from fpdf import FPDF

# Load configuration system
config = configparser.ConfigParser()
config.read('config.ini')  # Assuming configuration details are stored in config.ini

# Initialize logger
logging.basicConfig(filename='pytorch_settings.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_pytorch_settings() -> Dict[str, Any]:
    settings = {}
    try:
        settings['PyTorch Version'] = torch.__version__
        settings['CUDA Availability'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            settings['CUDA Version'] = torch.version.cuda
            settings['CUDA Device Count'] = torch.cuda.device_count()
            settings['CUDA Current Device'] = current_device
            settings['CUDA Device Name'] = torch.cuda.get_device_name(current_device)
            settings['CUDA Device Memory (GB)'] = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # Convert bytes to GB
    except Exception as e:
        logging.error(f"Error evaluating PyTorch settings: {e}")
    return settings

def log_evaluation_result(result: Dict[str, Any]) -> None:
    try:
        with open("pytorch_settings_log.txt", "a") as log_file:
            log_file.write("Evaluation Date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            for key, value in result.items():
                log_file.write(f"{key}: {value}\n")
            log_file.write("\n")
    except Exception as e:
        logging.error(f"Error logging evaluation result: {e}")

def send_email_notification(subject: str, body: str) -> None:
    sender_email = os.getenv('SENDER_EMAIL', config['EMAIL']['SENDER_EMAIL'])
    receiver_email = os.getenv('RECEIVER_EMAIL', config['EMAIL']['RECEIVER_EMAIL'])
    password = os.getenv('EMAIL_PASSWORD', config['EMAIL']['PASSWORD'])

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(config['EMAIL']['SMTP_SERVER'], int(config['EMAIL']['SMTP_PORT'])) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            logging.info("Email notification sent successfully.")
    except Exception as e:
        logging.error(f"Error sending email notification: {e}")

def upload_to_ftp(file_path: str) -> None:
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

def save_model_and_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, path: str) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, path)

def load_model_and_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> Tuple[int, float]:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def save_training_config(config_dict: Dict[str, Any], path: str) -> None:
    with open(path, 'w') as config_file:
        json.dump(config_dict, config_file)

def load_training_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as config_file:
        return json.load(config_file)

def generate_charts(result: Dict[str, Any], training_config: Dict[str, Any], epochs: int, losses: list) -> None:
    # PyTorch Settings Chart
    plt.figure(figsize=(10, 5))
    keys = list(result.keys())
    values = list(result.values())
    plt.barh(keys, values)
    plt.xlabel('Values')
    plt.ylabel('Settings')
    plt.title('PyTorch Settings Evaluation')
    plt.savefig('pytorch_settings_chart.png')
    plt.close()

    # Training Metrics Chart
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.savefig('training_metrics_chart.png')
    plt.close()

def generate_pdf_report(result: Dict[str, Any], training_config: Dict[str, Any], epochs: int, losses: list) -> None:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="PyTorch Settings Evaluation Report", ln=True, align='C')

    pdf.ln(10)
    pdf.cell(200, 10, txt="Evaluation Date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ln=True)
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="PyTorch Settings:", ln=True)
    for key, value in result.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Training Configuration:", ln=True)
    for key, value in training_config.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Training Metrics:", ln=True)
    for epoch, loss in enumerate(losses, 1):
        pdf.cell(200, 10, txt=f"Epoch {epoch}: Loss = {loss}", ln=True)

    pdf.add_page()
    pdf.image('pytorch_settings_chart.png', x=10, y=10, w=190)
    pdf.add_page()
    pdf.image('training_metrics_chart.png', x=10, y=10, w=190)

    pdf.output('pytorch_evaluation_report.pdf')

def backup_pytorch_settings():
    result = evaluate_pytorch_settings()
    logging.info("PyTorch Settings Evaluation Result:")
    for key, value in result.items():
        logging.info(f"{key}: {value}")
    log_evaluation_result(result)

def main():
    try:
        backup_pytorch_settings()

        # Dummy placeholders for model, optimizer, etc.
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        epochs = 10
        losses = [0.1 * (10 - i) for i in range(epochs)]

        save_model_and_optimizer(model, optimizer, epochs, losses[-1], 'checkpoint.pth')

        training_config = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'num_epochs': epochs,
        }
        save_training_config(training_config, 'config.json')

        generate_charts(evaluate_pytorch_settings(), training_config, epochs, losses)
        generate_pdf_report(evaluate_pytorch_settings(), training_config, epochs, losses)

        subject = "PyTorch Settings Backup Complete"
        body = "PyTorch settings have been successfully backed up."
        send_email_notification(subject, body)

        file_path = "pytorch_evaluation_report.pdf"
        if os.path.exists(file_path):
            upload_to_ftp(file_path)
        else:
            logging.error("Backup file not found.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
