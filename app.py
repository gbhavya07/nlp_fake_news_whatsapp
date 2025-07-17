from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import datetime
import pickle

# Initialize Flask and configure app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///messages.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Load the trained VotingClassifier
with open('voting_classifier.pkl', 'rb') as f:
    voting_clf = pickle.load(f)

# Load the trained TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"<User {self.username}>"

# Message Model
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(100), nullable=False)
    receiver = db.Column(db.String(100), nullable=False)
    message_text = db.Column(db.String(1000), nullable=False)
    timestamp = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), nullable=False)

    def __repr__(self):
        return f"<Message {self.sender} to {self.receiver}: {self.message_text}>"

# User Loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Add this block to ensure the tables are created on the first run
with app.app_context():
    db.create_all()

# Route for Home (Contacts)
@app.route("/")
@login_required
def index():
    # Fetch all other users as contacts for the logged-in user
    contacts = User.query.filter(User.id != current_user.id).all()
    return render_template("index.html", contacts=contacts, messages=[], active_contact=None)

# Route to send a message
@app.route("/send_message", methods=["POST"])
@login_required
def send_message():
    sender = current_user.username  # Get the sender as the logged-in user
    receiver_username = request.form['receiver']
    message_text = request.form['message_text']
    timestamp = str(datetime.datetime.now())

    # Predict if the message is FAKE or REAL
    status = predict_news(message_text)

    # Save the message to the database
    receiver = User.query.filter_by(username=receiver_username).first()
    if receiver:
        new_message = Message(sender=sender, receiver=receiver.username, message_text=message_text, timestamp=timestamp, status=status)
        db.session.add(new_message)
        db.session.commit()

    return redirect(url_for('chat', contact_id=receiver.id))

# Function for predicting message status
def predict_news(news_input):
    """
    Predict whether the input news is Fake or Real using the trained model.
    If the message has 2 or fewer words, classify it as Real directly.
    """
    # Split the input message into words and count them
    word_count = len(news_input.split())

    # If the message has 2 or fewer words, classify as "REAL" directly
    if word_count <= 2:
        return "REAL"
    
    # Otherwise, use the trained model to classify the message
    # Transform the input text using the TF-IDF vectorizer
    news_vectorized = tfidf.transform([news_input])  # Vectorize the input text
    
    # Predict using the VotingClassifier
    prediction = voting_clf.predict(news_vectorized)
    
    # Return "FAKE" for 0 and "REAL" for 1
    return "FAKE" if prediction[0] == 0 else "REAL"

# Route for chat
@app.route("/chat/<int:contact_id>")
@login_required
def chat(contact_id):
    # Get the active contact
    active_contact = User.query.get_or_404(contact_id)

    # Fetch messages exchanged between the logged-in user and the active contact
    messages = Message.query.filter(
        ((Message.sender == current_user.username) & (Message.receiver == active_contact.username)) |
        ((Message.sender == active_contact.username) & (Message.receiver == current_user.username))
    ).all()

    # Fetch all other users as contacts
    contacts = User.query.filter(User.id != current_user.id).all()

    return render_template("index.html", contacts=contacts, messages=messages, active_contact=active_contact)

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

# Signup route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already taken', 'danger')
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

# Logout route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

# Add this block to ensure the tables are created on the first run
with app.app_context():
    db.create_all()
 