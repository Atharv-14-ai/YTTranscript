import os
import re
import logging
import random
import random
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import yt_dlp
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# Database setup
DATABASE_URL = "postgresql://postgres:new_password@localhost:5432/YT"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
db_session = Session()
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False)
    email = Column(String(120), unique=True, nullable=False)

    notes = relationship("Note", back_populates="user")
    interactions = relationship("UserInteraction", back_populates="user")
    feedbacks = relationship("Feedback", back_populates="user")

class Note(Base):
    __tablename__ = 'notes'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="notes")

class Video(Base):
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True)
    youtube_link = Column(String(255), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(String)
    category = Column(String(100))

    interactions = relationship("UserInteraction", back_populates="video")

class UserInteraction(Base):
    __tablename__ = 'user_interactions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    action = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="interactions")
    video = relationship("Video", back_populates="interactions")

class Feedback(Base):
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    rating = Column(Integer)  # Assuming rating is an integer between 1-5
    comment = Column(Text)

    user = relationship("User", back_populates="feedbacks")

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = 'your_secret_key'
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt_template = """You are a YouTube video summarizer. You will take the transcript text
and summarize the entire video, providing important points within {length} words.
The transcript text is as follows: """

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        new_user = User(username=username, email=email)
        
        db_session.add(new_user)
        db_session.commit()
        
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        user = db_session.query(User).filter_by(email=email).first()
        if user:
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid email or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/youtube_transcript_summarizer', methods=['GET', 'POST'])
def youtube_transcript_summarizer():
    summary = ""
    pdf_file_path = ""
    error_message = ""
    youtube_link = ""  # Initialize variable to hold YouTube link

    if request.method == 'POST':
        youtube_link = request.form['youtube_link']  # Get the YouTube link from form
        summary_length = request.form['summary_length']
        
        if not re.match(r'^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$', youtube_link):
            error_message = "Please enter a valid YouTube link."
        else:
            length_options = {
                'Short (100 words)': 100,
                'Medium (300 words)': 300,
                'Long (500 words)': 500,
            }
            length = length_options.get(summary_length, 100)  # Default to short if not found
            
            transcript_text, error_message = extract_transcript_details(youtube_link)

            if transcript_text:
                summary = generate_summary(transcript_text, length)
                video_id_match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", youtube_link)
                youtube_video_id = video_id_match.group(1) if video_id_match else None
                
                video = db_session.query(Video).filter_by(youtube_link=youtube_link).first()
                
                if not video:
                    new_video = Video(youtube_link=youtube_link, title="", description="", category="")
                    db_session.add(new_video)
                    db_session.commit()
                    video_id = new_video.id 
                else:
                    video_id = video.id

                pdf_file_name = save_as_pdf(summary, youtube_video_id)

                user_id=session.get('user_id')
                if user_id:
                    log_user_interaction(user_id ,video_id ,'summarized')

                pdf_file_path=os.path.join(os.getcwd(), 'downloads', pdf_file_name)

    return render_template('index.html', summary=summary,
                           pdf_file=pdf_file_path,
                           error_message=error_message,
                           youtube_link=youtube_link)  # Pass the YouTube link back to the template

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    questions = []
    feedback = ""

    if request.method == 'POST':
        # Get user answers from the form
        user_answers = request.form.getlist('answers')
        
        # Assuming correct answers are stored in a way you can access
        correct_answers = [question['correct_answer'] for question in questions]

        # Check answers and provide feedback
        score = sum(1 for i in range(len(user_answers)) if user_answers[i] == correct_answers[i])
        total_questions = len(correct_answers)
        feedback = f"You scored {score} out of {total_questions}."

        # Redirect to results page with score as a URL parameter
        return redirect(url_for('quiz_result', score=score, total=total_questions))

    # Generate questions based on the summary or transcript
    summary = request.args.get('summary', '')
    questions = generate_quiz_questions(summary)

    return render_template('quiz.html', questions=questions, feedback=feedback)

@app.route('/result')
def quiz_result():
    score = request.args.get('score', type=int)
    total_questions = request.args.get('total', type=int)
    return render_template('quiz_result.html', score=score, total=total_questions)

def generate_quiz_questions(summary):
    sentences = summary.split('. ')
    questions = []

    for sentence in sentences:
        if sentence:
            question_text = f"What is the main idea of this statement? '{sentence}'"
            
            # Generate plausible incorrect options
            similar_statements = [
                f"This statement suggests that {sentence.split()[0]} is crucial.",
                f"Many believe that {sentence.split()[1]} is often misunderstood.",
                f"This implies that {sentence.split()[2]} has significant implications.",
                f"{sentence.split()[3]} plays a vital role in this context."
                # Add more variations as needed
            ]
            
            options = [sentence] + random.sample(similar_statements, k=3) # Ensure we have enough options
            
            random.shuffle(options)  
            
            questions.append({
                'question': question_text,
                'options': options,
                'correct_answer': sentence  
            })
    
    return questions
    

@app.route('/video_downloader', methods=['GET', 'POST'])
def video_downloader():
    download_message=""
    error_message=""
    
    if request.method == 'POST':
        video_link=request.form['video_link']
        video_quality=request.form['video_quality']  # Get selected video quality
        
        # Validate YouTube link format
        if not re.match(r'^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$',video_link):
            error_message="Please enter a valid YouTube video link."
        else:
            try:
                download_dir=os.path.join(os.getcwd(),'downloads')
                os.makedirs(download_dir ,exist_ok=True)  # Create downloads directory if it doesn't exist
                
                ydl_opts={
                    'format': f'bestvideo[height={video_quality}]+bestaudio/best',
                    'outtmpl': os.path.join(download_dir,'% (title)s.%(ext)s'),
                    'merge_output_format': 'mp4'  # Merge audio and video into MP4 format
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_link])
                
                download_message += "Video downloaded successfully!"

                # Log the interaction (if user is logged in)
                user_id=session.get('user_id')
                if user_id:
                    video_id_match=re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})",video_link)
                    video_id=video_id_match.group(1)if(video_id_match)else None 
                    log_user_interaction(user_id ,video_id ,'downloaded')

            except Exception as e:
                logging.error(f"Error downloading video: {e}")
                error_message=f"An error occurred while downloading the video: {str(e)}"

    return render_template('video_downloader.html',
                           download_message=download_message,
                           error_message=error_message)

@app.route('/mp3_downloader', methods=['GET', 'POST'])
def mp3_downloader():
    download_message = ""
    error_message = ""
    
    if request.method == 'POST':
        video_link = request.form['video_link']  # Ensure this matches the input name in your HTML
        
        # Validate YouTube link format
        if not re.match(r'^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$', video_link):
            error_message = "Please enter a valid YouTube video link."
        else:
            try:
                download_dir = os.path.join(os.getcwd(), 'downloads')
                os.makedirs(download_dir, exist_ok=True)  # Create downloads directory if it doesn't exist
                
                ydl_opts = {
                    'format': 'bestaudio/best',  # Download the best audio quality
                    'extractaudio': True,  # Download only audio
                    'audioformat': 'mp3',  # Save as mp3
                    'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),  # Save to downloads directory 
                    'noplaylist': True  # Prevent downloading playlists 
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(video_link, download=False)
                    ydl.download([video_link])  # Download the audio

                download_message += "Audio downloaded successfully!"

                # Log the interaction (if user is logged in)
                user_id = session.get('user_id')
                if user_id:
                    video_id_match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", video_link)
                    video_id = video_id_match.group(1) if video_id_match else None 
                    log_user_interaction(user_id, video_id, 'downloaded')

            except Exception as e:
                logging.error(f"Error downloading audio: {e}")
                error_message = f"An error occurred while downloading the audio: {str(e)}"

    return render_template('mp3_downloader.html', download_message=download_message, error_message=error_message)

def extract_transcript_details(youtube_video_url):
     video_id_match=re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})",youtube_video_url)

     if not(video_id_match):
         return None,"Invalid YouTube URL. Please enter a valid link."

     video_id=video_id_match.group(1)

     try:
         transcript_text=YouTubeTranscriptApi.get_transcript(video_id)
         transcript=" ".join([i["text"] for i in transcript_text])
         return transcript,None

     except NoTranscriptFound:
         return None,"No transcript found for this video."
     except Exception as e:
         logging.error(f"Error fetching transcript: {e}")
         return None,"An error occurred while fetching the transcript."

def generate_summary(transcript_text,length):
      model=genai.GenerativeModel("gemini-pro")
      prompt=prompt_template.format(length=length)
      response=model.generate_content(prompt+transcript_text)
      return response.text

def save_as_pdf(summary_text ,video_id):
      styles=getSampleStyleSheet()
      heading_style=ParagraphStyle(name='Heading',
                                   fontSize=16,
                                   leading=20,
                                   spaceAfter=12,
                                   fontName='Helvetica-Bold')
      body_style=ParagraphStyle(name='Body',
                                fontSize=12,
                                leading=16,
                                spaceAfter=6,
                                alignment=4)

      elements=[]
      elements.append(Paragraph('Summary',heading_style))
      elements.append(Spacer(1 ,12))

      paragraphs=summary_text.split('\n')

      for para in paragraphs:
          elements.append(Paragraph(para ,body_style))
          elements.append(Spacer(1 ,6))

      filename=f"NotesWaves.Ai{video_id}.pdf"

      doc_path=os.path.join(os.getcwd() ,'downloads' ,filename)

      doc=SimpleDocTemplate(doc_path,
                            pagesize=letter,
                            leftMargin=36,
                            rightMargin=36,
                            topMargin=36,
                            bottomMargin=18)

      doc.build(elements)

      logging.info(f"PDF saved as {filename}")

      return filename

@app.route('/feedback', methods=['POST'])
def feedback():
   user_id=session.get('user_id')

   if user_id and request.method == 'POST':
       video_id=request.form.get('video_id','').strip()  # Get video ID safely
    
       rating=request.form.get('rating','').strip()  # Get rating safely
       
       comment=request.form.get('comment','')

       if not rating.isdigit(): 
           error_message="Please provide a valid rating."
           return render_template('index.html',error_message=error_message) 

       new_feedback=Feedback(user_id=user_id ,video_id=video_id,
                             rating=int(rating),comment=comment)

       db_session.add(new_feedback)
       db_session.commit()

       return redirect(url_for('some_page')) 

   return render_template('index.html',
                          error_message="You must be logged in to submit feedback.")

def log_user_interaction(user_id ,video_id ,action):
   new_interaction=UserInteraction(user_id=user_id ,
                                    video_id=video_id ,
                                    action=action)

   db_session.add(new_interaction) 
   db_session.commit()

@app.route('/download/<filename>')
def download_file(filename):
   file_path=os.path.join(os.getcwd() ,'downloads' ,filename)

   if os.path.exists(file_path):
       return send_file(file_path ,as_attachment=True)

   else:
       return "file downloaded",404

# Route for generating notes from study material (to be implemented later)
@app.route('/generate_notes', methods=['GET', 'POST'])
def generate_notes():
    note_content = ""
    summary = ""
    error_message = ""
    pdf_file_path = ""

    if request.method == 'POST':
        note_content = request.form['note_content']
        
        # Use AI to generate notes or summaries from provided content
        summary = create_notes(note_content)  # Generate notes
        
        # Save the generated notes as a PDF
        pdf_filename = f"NotesWaves.AI{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_file_path = save_notes_as_pdf(summary, pdf_filename)

    return render_template('generate_notes.html', note_content=note_content,
                           summary=summary,
                           
                           pdf_file_path=pdf_file_path)  # Pass the path for download


def save_notes_as_pdf(content, filename):

    # Create a path for saving the PDF
    doc_path = os.path.join('downloads', filename)
    
    # Ensure directory exists
    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    styles = getSampleStyleSheet()
    
    # Define custom styles for headings and body text
    heading_style = ParagraphStyle(name='Heading',
                                   fontSize=16,
                                   leading=20,
                                   spaceAfter=12,
                                   fontName='Helvetica-Bold')
    
    body_style = ParagraphStyle(name='Body',
                                fontSize=12,
                                leading=16,
                                spaceAfter=6,
                                alignment=4)

    doc = SimpleDocTemplate(doc_path, pagesize=letter)

    elements = []

    # Add a title to the PDF
    title = Paragraph("Generated Study Notes", heading_style)
    elements.append(title)
    elements.append(Spacer(1, 12))  # Add space after title

    # Split content into paragraphs
    paragraphs = content.split('<br>')

    for para in paragraphs:
        if para.strip():  # Avoid adding empty paragraphs
            if para.startswith('- '):  # Check if it's a bullet point (you can customize this logic)
                bullet_points = para.split('- ')[1:]  # Split by bullet point indicator
                bullet_list = ListFlowable(
                    [ListItem(Paragraph(bp.strip(), body_style), bulletType='bullet') for bp in bullet_points],
                    bulletType='bullet'
                )
                elements.append(bullet_list)
            else:
                elements.append(Paragraph(para.strip(), body_style))  # Add each paragraph
            elements.append(Spacer(1, 6))  # Add space between paragraphs

    # Build the PDF
    doc.build(elements)

    return doc_path  # Return the path to the saved PDF

def upload_pdf():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            return render_template('upload_pdf.html', error="No file part")
        
        file = request.files['pdf_file']
        
        if file.filename == '':
            return render_template('upload_pdf.html', error="No selected file")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)  # Ensure 'uploads' directory exists
            os.makedirs('uploads', exist_ok=True)
            file.save(file_path)

            # Read the PDF content
            pdf_text = extract_text_from_pdf(file_path)

            # Optionally generate notes from the PDF content here using AI
            summary = create_notes(pdf_text)

            return render_template('upload_pdf.html', summary=summary)

    return render_template('upload_pdf.html')

def allowed_file(filename):
     return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

def extract_text_from_pdf(file_path):
     text=""
     with open(file_path,"rb") as file:
         reader=PdfReader(file)
         for page in reader.pages:
             text+=page.extract_text()+"\n"  # Extract text from each page

     return text

def create_notes(content):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Generate concise study notes from the following content:\n\n{content}"
    response = model.generate_content(prompt)
    
    # Clean up the output by removing unwanted characters (like asterisks)
    formatted_notes = response.text.replace('*', '').replace('\n', '<br>')  # Replace newlines with <br> for HTML line breaks
    return formatted_notes


@app.route('/delete_all_data', methods=['POST'])
def delete_all_data():
    try:
        # Delete all records from each table
        db_session.query(User).delete()
        db_session.query(Note).delete()
        db_session.query(Video).delete()
        db_session.query(UserInteraction).delete()
        db_session.query(Feedback).delete()
        
        # Commit the changes
        db_session.commit()
        
        return "All data has been successfully deleted."
    except Exception as e:
        db_session.rollback()  # Rollback in case of error
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
   app.run(debug=True)