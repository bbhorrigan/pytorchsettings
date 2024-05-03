import torch
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from ftplib import FTP

# Function to evaluate PyTorch settings
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
        print(f"Error evaluating PyTorch settings: {e}")
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
        print(f"Error logging evaluation result: {e}")

# Function to send email notification
def send_email_notification(subject, body):
    sender_email = "your_email@gmail.com"  # Update with your email
    receiver_email = "recipient_email@example.com"  # Update with recipient's email
    password = "your_email_password"  # Update with your email password

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email notification sent successfully.")
    except Exception as e:
        print(f"Error sending email notification: {e}")
    finally:
        server.quit()

# Function to upload backup file to FTP server
def upload_to_ftp(file_path):
    ftp_host = 'ftp.example.com'  # Update with FTP host
    ftp_username = 'ftp_username'  # Update with FTP username
    ftp_password = 'ftp_password'  # Update with FTP password

    try:
        with FTP(ftp_host) as ftp:
            ftp.login(user=ftp_username, passwd=ftp_password)
            with open(file_path, 'rb') as f:
                ftp.storbinary(f'STOR {os.path.basename(file_path)}', f)
        print("File uploaded to FTP server successfully.")
    except Exception as e:
        print(f"Error uploading file to FTP server: {e}")

# Main function
def main():
    try:
        result = evaluate_pytorch_settings()
        print("PyTorch Settings Evaluation Result:")
        for key, value in result.items():
            print(f"{key}: {value}")
        log_evaluation_result(result)

        # Send email notification
        subject = "PyTorch Settings Backup Complete"
        body = "PyTorch settings have been successfully backed up."
        send_email_notification(subject, body)

        # Upload backup file to FTP server
        file_path = "pytorch_settings_log.txt"
        if os.path.exists(file_path):
            upload_to_ftp(file_path)
        else:
            print("Backup file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
