from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
import joblib
import pandas as pd
import json
import os
import google.generativeai as genai
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import logging
import time

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure random secret key

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDJ7bLKE-doWzFY1OEWD43iORRhCAPEzEQ"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Load the best saved model
try:
    model = joblib.load('best_model.pkl')
    logger.info("Loaded best model from 'best_model.pkl'.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

# Define the 10 feature names used in the model
features = ['Machine', 'DebugSize', 'MajorImageVersion', 'ExportSize',
            'IatVRA', 'NumberOfSections', 'SizeOfStackReserve',
            'DllCharacteristics', 'ResourceSize', 'BitcoinAddresses']

# Helper function for retrying Gemini requests
def try_gemini_request(prompt, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = gemini_model.generate_content(prompt)
            logger.debug(f"Gemini response (attempt {attempt + 1}): {response.text}")
            if response.text:
                return response.text
            logger.warning(f"Gemini response empty on attempt {attempt + 1}")
        except Exception as e:
            logger.error(f"Gemini request failed (attempt {attempt + 1}): {str(e)}")
        if attempt < retries - 1:
            time.sleep(delay)
    return None

# Helper functions for JSON user database
def load_users():
    if os.path.exists('users.json'):
        try:
            with open('users.json', 'r') as f:
                data = json.load(f)
                if "users" in data:
                    return data
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error loading users.json: {str(e)}")
    # Overwrite with valid structure if error
    save_users({"users": []})
    return {"users": []}

def save_users(users):
    try:
        with open('users.json', 'w') as f:
            json.dump(users, f, indent=4)
        logger.info("Users saved to users.json.")
    except Exception as e:
        logger.error(f"Error saving users.json: {str(e)}")

# Simple decorator to protect routes that require login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Decorator for admin-only routes
def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        users = load_users()
        user = next((u for u in users['users'] if u['username'] == session['username']), None)
        if not user or not user.get('is_admin', False):
            flash("Admin access required.", "danger")
            return redirect(url_for('predict'))
        return f(*args, **kwargs)
    return decorated_function

# Landing page (no login required)
@app.route('/')
def landing():
    return render_template('landing.html')

# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')

        if not all([username, password, email]):
            flash("All fields are required.", "danger")
            return render_template('register.html')

        users = load_users()
        for user in users["users"]:
            if user["username"] == username:
                flash("Username already exists.", "danger")
                return render_template('register.html')
            if user.get("email") == email:
                flash("Email already registered.", "danger")
                return render_template('register.html')

        new_user = {
            "username": username,
            "password": generate_password_hash(password),
            "email": email,
            "is_admin": False  # Default: not admin
        }
        users["users"].append(new_user)
        save_users(users)
        flash("Registration successful. Please log in.", "success")
        logger.info(f"New user registered: {username}")
        return redirect(url_for('login'))
    return render_template('register.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not all([username, password]):
            flash("All fields are required.", "danger")
            return render_template('login.html')

        users = load_users()
        for user in users["users"]:
            if user["username"] == username and check_password_hash(user["password"], password):
                session['username'] = username
                session['email'] = user['email']  # ✅ Store email in session
                session['is_admin'] = user.get('is_admin', False)
                
                flash("Login successful.", "success")
                logger.info(f"User logged in: {username} ({user['email']})")
                return redirect(url_for('predict'))

        flash("Invalid username or password.", "danger")
        logger.warning(f"Failed login attempt for username: {username}")
        return render_template('login.html')

    return render_template('login.html')


# Route for user logout
@app.route('/logout')
def logout():
    username = session.get('username', 'Unknown')
    session.pop('username', None)
    flash("Logged out successfully.", "success")
    logger.info(f"User logged out: {username}")
    return redirect(url_for('login'))

import smtplib
from email.message import EmailMessage

def send_prediction_email(to_email, subject, body):
    try:
        print(f"[DEBUG] Preparing to send email to: {to_email}")  # Print the email address

        email_address = 'daminmain@gmail.com'
        email_password = 'kpqtxqskedcykwjz'  # App password, not real password!

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = email_address
        msg['To'] = to_email
        msg.set_content(body)

        print("[DEBUG] Connecting to SMTP server...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_address, email_password)
            print("[DEBUG] Logged in to SMTP server.")
            smtp.send_message(msg)
            print("[DEBUG] Email sent successfully.")

        logger.info(f"Prediction result sent to {to_email}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {str(e)}")
        logger.error(f"Failed to send email: {str(e)}")


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            sample_data = {}
            for key in features:
                value = request.form.get(key)
                if value is None:
                    raise ValueError(f"Missing field: {key}")
                value = int(value)
                if value < 0:
                    raise ValueError(f"{key} cannot be negative")
                if key == 'NumberOfSections' and value == 0:
                    raise ValueError("NumberOfSections must be positive")
                sample_data[key] = value

            sample_df = pd.DataFrame([sample_data])

            if model is None:
                flash("Model not loaded. Contact admin.", "danger")
                logger.error("Prediction attempted with no model loaded.")
                return redirect(url_for('predict'))

            prediction = model.predict(sample_df)[0]
            try:
                probabilities = model.predict_proba(sample_df)[0]
                confidence = max(probabilities)
            except AttributeError:
                confidence = None

            label_mapping = {1: "Benign", 0: "Malicious"}
            predicted_label = label_mapping.get(prediction, "Unknown")

            feature_contributions = None
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_contributions = {f: imp for f, imp in zip(features, importances)}

            input_text = (
                f"The model predicted '{predicted_label}' with {confidence * 100:.2f}% confidence. "
                f"Input features: {sample_data}. "
                f"Key features influencing the prediction: {feature_contributions or 'N/A'}. "
                f"If malicious, suggest specific ransomware mitigation steps (e.g., isolation, scanning). "
                f"If benign, recommend monitoring strategies to ensure safety."
            )

            try:
                gpt_response = try_gemini_request(input_text, retries=3, delay=2)
                if not gpt_response:
                    gpt_response = "No analysis generated. Please try again later."
                    logger.warning("All Gemini request attempts failed or returned empty response.")
            except Exception as e:
                gpt_response = f"Error generating analysis: {str(e)}"
                logger.error(f"Unexpected Gemini error: {str(e)}")

            # ✅ Send email with debug print
            user_email = session.get('email')
            print(f"[DEBUG] User email from session: {user_email}")
            if user_email:
                email_subject = f"Prediction Result for {session.get('username', 'User')}"
                email_body = (
                    f"Hello {session.get('username', 'User')},\n\n"
                    f"Prediction: {predicted_label}\n"
                    f"Confidence: {confidence * 100:.2f}%\n\n"
                    f"Gemini Analysis:\n{gpt_response}\n\n"
                    f"Input Features:\n{sample_data}\n\n"
                    f"Feature Contributions:\n{feature_contributions or 'N/A'}\n\n"
                    f"- This is an automated message from the prediction system."
                )
                send_prediction_email(user_email, email_subject, email_body)

            session['prediction_data'] = {
                'prediction': predicted_label,
                'confidence': f"{confidence * 100:.2f}%" if confidence else "N/A",
                'gpt_response': gpt_response,
                'input_data': sample_data,
                'feature_contributions': feature_contributions
            }

            logger.info(f"Prediction made for user {session['username']}: {predicted_label}")
            return render_template('result.html',
                                   prediction=predicted_label,
                                   confidence=confidence,
                                   gpt_response=gpt_response,
                                   feature_contributions=feature_contributions)

        except (ValueError, KeyError) as e:
            flash(f"Invalid input: {str(e)}", "danger")
            logger.error(f"Prediction error: {str(e)}")
            return render_template('predict.html')
        except Exception as e:
            flash("An unexpected error occurred during prediction.", "danger")
            logger.error(f"Unexpected prediction error: {str(e)}")
            return render_template('predict.html')

    return render_template('predict.html')


# Route for downloading the report
@app.route('/download_report')
@login_required
def download_report():
    prediction_data = session.get('prediction_data', {})
    if not prediction_data:
        flash("No report available to download.", "warning")
        logger.warning(f"User {session['username']} attempted to download report with no prediction data.")
        return redirect(url_for('predict'))

    # Create PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("RansomGuard Threat Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Prediction
    story.append(Paragraph("Prediction Result", styles['Heading2']))
    story.append(Paragraph(f"Predicted Class: {prediction_data['prediction']}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Confidence
    story.append(Paragraph(f"Confidence Score: {prediction_data['confidence']}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Input Data
    story.append(Paragraph("Input Features:", styles['Heading2']))
    for key, value in prediction_data['input_data'].items():
        story.append(Paragraph(f"{key}: {value}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Feature Contributions
    if prediction_data.get('feature_contributions'):
        story.append(Paragraph("Feature Contributions:", styles['Heading2']))
        for key, value in prediction_data['feature_contributions'].items():
            story.append(Paragraph(f"{key}: {value:.4f}", styles['BodyText']))
        story.append(Spacer(1, 12))

    # AI Analysis
    story.append(Paragraph("AI Analysis:", styles['Heading2']))
    story.append(Paragraph(prediction_data['gpt_response'], styles['BodyText']))

    # Build PDF
    try:
        doc.build(story)
        buffer.seek(0)
        logger.info(f"Report downloaded by user {session['username']}.")
        return send_file(buffer, as_attachment=True, download_name="RansomGuard_Report.pdf", mimetype='application/pdf')
    except Exception as e:
        flash("Error generating report.", "danger")
        logger.error(f"Report generation error: {str(e)}")
        return redirect(url_for('predict'))

# Chat route
@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if not user_input:
            return jsonify({'response': 'Input required.'})

        chatbot_input = (
            f"User: {user_input}\n"
            f"Chatbot: Provide information and solutions related to ransomware attacks based on the user's query. "
            f"Focus strictly on ransomware topics."
        )
        try:
            gpt_response = try_gemini_request(chatbot_input, retries=3, delay=2)
            if not gpt_response:
                gpt_response = "No response generated. Please try again later."
                logger.warning("All Gemini request attempts failed or returned empty response.")
            else:
                logger.info(f"Chat response generated for user {session['username']}.")
            return jsonify({'response': gpt_response})
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return jsonify({'response': f"Error: {str(e)}"})
    return render_template('chat.html')

# Route for retraining the model (admin only)
@app.route('/retrain', methods=['GET', 'POST'])
@admin_required
def retrain():
    if request.method == 'POST':
        try:
            # Run the training script
            from train_model import train_model
            train_model()
            # Reload the model
            global model
            model = joblib.load('best_model.pkl')
            flash("Model retrained successfully.", "success")
            logger.info(f"Model retrained by admin {session['username']}.")
        except Exception as e:
            flash(f"Error retraining model: {str(e)}", "danger")
            logger.error(f"Retraining error: {str(e)}")
        return redirect(url_for('retrain'))
    return render_template('retrain.html')

if __name__ == '__main__':
    app.run(debug=True)