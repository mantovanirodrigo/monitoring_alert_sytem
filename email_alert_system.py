# -*- coding: utf-8 -*-
import asyncio 

async def send_email_async(feature, timestamp, actual_value, predicted_value, threshold):
    
    from email.message import EmailMessage
    import ssl
    import smtplib
    import asyncio
    
    sender_email = "anomalies.alert.system@gmail.com"
    receiver_email = "anomalies.alert.system@gmail.com"
    password = "klcu cnyt cwic odun"

    # Email configuration
    
    subject = '[M.A.S.] Abnormal transaction behavior identified'
    body = f"""
    
    ***** ANOMALY ALERT *****
    
    The monitoring system identified abnormal behavior in the relative amounts of {feature} transactions.
    
    Here is the complete information about the anomaly:
        
    - Time: {timestamp}
    - % of failed transactions: {(100*actual_value):.2f}%
    - Predicted value for % of failed transactions: {(100*predicted_value):.2f}%
    - Exceeded current threshold ({(100*threshold):.2f}%) in {(100*(actual_value - predicted_value - threshold)):.2f}
    
    """
    
    em = EmailMessage()
    em['From'] = sender_email
    em['To'] = receiver_email
    em['Subject'] = subject
    em.set_content(body)
   
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context = context) as smtp:
        
        smtp.login(sender_email, password)
        smtp.sendmail(sender_email, receiver_email,
                      em.as_string())
        
def send_email(feature, timestamp, actual_value, predicted_value, threshold):
    asyncio.run(send_email_async(feature, timestamp, actual_value, predicted_value, threshold))
