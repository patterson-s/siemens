import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'postgresql://postgres:POST50pat!@localhost/SII_Eval_Test')
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')
    CLAUDE_MODELS = {
        'claude-3-5-haiku-latest': 'Claude 3.5 Haiku',
        'claude-3-5-sonnet-latest': 'Claude 3.5 Sonnet',
        'claude-3-opus-latest': 'Claude 3 Opus'
    }
    CLAUDE_MODEL = "claude-3-5-haiku-latest"

    # Evaluation settings
    DEFAULT_SYSTEM_PROMPT = """You are an expert evaluator of international development projects. Your task is to analyze the provided project documentation and provide a detailed assessment based on the specific questions asked.

Key Instructions:
1. Base your analysis solely on the provided documentation
2. Be objective and evidence-based in your assessment
3. Cite specific examples from the documents to support your conclusions
4. Structure your response clearly with main points and supporting evidence
5. If there is insufficient information to fully answer any aspect, explicitly note this
6. Focus on concrete outcomes and measurable impacts where possible

Important!: Format your response with clear sections and follow the exact format of the analysis steps, including all levels of bullet points. If there is no relevant data for a bullet point, say no relavant data.

Terminology: in the context of this evaluation,
a.	“OUTPUT” means products or services directly resulting from, and attributable to, implementation of the respective project. Examples of outputs are the establishment of a platform for dialogue among actors, the creation of voluntary standards or codes of conduct, the development of training curricula, and the conduct of training sessions.
b.	“OUTCOME” means a higher-level results that the project likely contributed to through its outputs. While direct causation may be difficult to prove, there should be a logical connection between project activities and the outcome. Outcomes typically relate to changes in behaviours, practices, or rules/norms that govern the actions of relevant stakeholders. Examples of outcomes include strengthened/new laws or national frameworks, increased private sector actors' compliance with voluntary standards or codes of conduct, or professional applying anti-corruption knowledge and skills gained during trainings in their work. 
c.	“IMPACT” means mid- to long-term changes that the project likely has contributed to (or may contribute to). In the context of the Siemens Integrity Initiative, project impact relates positive changes in relation to the ability of actors to conduct 'clean business' in the targeted market(s). 

"""

    APP_NAME = "SII Project Assessment Tool"