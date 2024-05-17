import torch
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from ftplib import FTP

# Function to import PyTorch settings from backup file, update the value of the files to the location on your machine
def import_pytorch_settings(file_path):
    settings = {}
    try:
        with open(file_path, "r") as backup_file:
            lines = backup_file.readlines()
            for line in lines:
                if ":" in line:
                    key, value = line.split(":")
                    settings[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error importing PyTorch settings: {e}")
    return settings

# Function to send email notification, change settings to match needs, comment out the rest.
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
        # Import PyTorch settings from backup file
        backup_file_path = "pytorch_settings_backup.txt"
        imported_settings = import_pytorch_settings(backup_file_path)
        print("PyTorch Settings Imported Successfully:")
        for key, value in imported_settings.items():
            print(f"{key}: {value}")

        # Send email notification
        subject = "PyTorch Settings Import Complete"
        body = "PyTorch settings have been successfully imported."
        send_email_notification(subject, body)

        # Upload backup file to FTP server
        if os.path.exists(backup_file_path):
            upload_to_ftp(backup_file_path)
        else:
            print("Backup file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
