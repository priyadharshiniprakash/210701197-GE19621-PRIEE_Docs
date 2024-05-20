import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import PyPDF2
load_dotenv()

# Load job dataset
def load_job_data(filepath):
    return pd.read_csv(filepath)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():

    # Set up the customization options
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )

    llm = ChatGroq(
            temperature=0, 
            groq_api_key=os.environ.get("GROQ_API_KEY"), 
            model_name=model
        )

    # Load job dataset
    job_data = load_job_data('jobs.csv')

    # Streamlit UI
    st.title('Job Recommender')
    multiline_text = """
    Job Recommender is a tool that helps you find the right job based on your preferences and skills. Upload your resume now to get started!
    """

    st.markdown(multiline_text, unsafe_allow_html=True)
    
    # Conversational prompts
    st.write("Hello! I'm excited to help you find your next job opportunity. To get started, I'd like to understand your job preferences and qualifications.")
    st.write("Could you please share your resume with me? This will help me better understand your skills and experience.")
    
    user_question = st.text_input("What type of a job are you looking for? Are you looking for a specific industry, company, or location? Do you have any specific skills or qualifications that you'd like to highlight?")
    uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX)")

    if uploaded_file is not None:
        st.success("Resume uploaded successfully.")
        resume_text = extract_text_from_pdf(uploaded_file)
        
    if user_question and uploaded_file is not None:
        st.write("Great! Let me process your resume and find some job opportunities for you.")
        
        Resume_Parsing_Agent = Agent(
            role='Resume_Parsing_Agent',
            goal="""Analyze the user's resume to extract relevant information such as skills, experience, and qualifications. Find out their most recent job title and the skills they have.""",
            backstory="""You are an expert in understanding and analyzing resumes. 
            Your goal is to extract the most relevant information from the user's resume and use it to recommend suitable job roles.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )
        
        Job_Finder_Agent = Agent(
            role='Job_Finder_Agent',
            goal="""Search and identify job opportunities that match the user's qualifications and preferences based on the extracted resume data and specified job criteria.""",
            backstory="""You have extensive knowledge of job databases and search algorithms. 
            Your mission is to find and recommend job listings that best align with the user's skills, experience, and job preferences.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )
        
        User_Interaction_Agent = Agent(
            role='User_Interaction_Agent',
            goal="""Engage with the user to gather their job preferences, assist with the resume upload process, and present matched job opportunities. Provide support and answer any questions the user may have.""",
            backstory="""You are a friendly and knowledgeable assistant dedicated to helping users navigate the job search process. 
            Your primary objective is to ensure users have a smooth and efficient experience finding job opportunities.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        # Define tasks
        task_parse_resume = Task(
            description=f"""Parse the user's uploaded resume to extract skills, experience, and qualifications.
                            Resume text: {resume_text}""",
            agent=Resume_Parsing_Agent,
            expected_output="Extracted skills, experience, and qualifications from the resume."
        )

        task_find_jobs = Task(
            description=f"""Find job listings from the job dataset that match the user's extracted resume data and specified job preferences.
                            Here is the job dataset: {job_data}""",
            agent=Job_Finder_Agent,
            expected_output="A list of job opportunities that match the user's qualifications and preferences."
        )

        task_interact_with_user = Task(
            description="""Interact with the user to understand their job preferences and present the matched job opportunities.""",
            agent=User_Interaction_Agent,
            expected_output="Engaged user with relevant job listings and provided support as needed."
        )

        crew = Crew(
            agents=[Resume_Parsing_Agent, Job_Finder_Agent, User_Interaction_Agent],
            tasks=[task_parse_resume, task_find_jobs, task_interact_with_user],
            verbose=2
        )

        result = crew.kickoff()

        # Displaying the results in Streamlit UI
        st.subheader("Job Recommendations:")
        st.write("Based on your resume and preferences, here are some job opportunities that might interest you:")
        st.write(result)

if __name__ == "__main__":
    main()
