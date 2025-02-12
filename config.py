import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# Load .env file only in development
if os.environ.get('RENDER') is None:
    load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    
    # App configuration
    APP_NAME = "SII AI Assessment Tool v 1.1"
    
    # Database configuration
    IS_PRODUCTION = os.environ.get('RENDER') is not None
    print(f"Environment: {'Production' if IS_PRODUCTION else 'Development'}")
    
    if IS_PRODUCTION:
        # Production database URL (from Render)
        DATABASE_URL = os.environ.get('DATABASE_URL')
        print(f"Production DATABASE_URL: {DATABASE_URL}")
        if DATABASE_URL:
            SQLALCHEMY_DATABASE_URI = DATABASE_URL.replace("postgres://", "postgresql://", 1)
            print(f"Using Production DB: {SQLALCHEMY_DATABASE_URI}")
    else:
        # Local development database URL
        DATABASE_URL = 'postgresql://postgres:POST50pat!@localhost/SII_Eval_Test'
        SQLALCHEMY_DATABASE_URI = DATABASE_URL
        print(f"Using Development DB: {SQLALCHEMY_DATABASE_URI}")
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')
    CLAUDE_MODELS = {
        'claude-3-5-haiku-latest': 'Claude 3.5 Haiku',
        'claude-3-5-sonnet-latest': 'Claude 3.5 Sonnet',
        'claude-3-opus-20240229': 'Claude 3.0 Opus'
    }
    CLAUDE_MODEL = "claude-3-5-haiku-latest"

    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


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

There may be two document provided.  One will be of the type "external_evaluation" and the other "final_project_report". The final project report was prepared by the organization running the project and will have information and details according to their perspective. The external evaluation is an evaluation of the final project report by an external evaluator and will have information and details according to that perspective. To answer the questions in this assessment process, you should give the greatest weight to the external evaluation and use the final project report to provide addiitional details that may be relevant but which were not present in the external evaluation. .


Terminology: in the context of this evaluation,
a.	“OUTPUT” means products or services directly resulting from, and attributable to, implementation of the respective project. Examples of outputs are the establishment of a platform for dialogue among actors, the creation of voluntary standards or codes of conduct, the development of training curricula, and the conduct of training sessions.
b.	“OUTCOME” means a higher-level results that the project likely contributed to through its outputs. While direct causation may be difficult to prove, there should be a logical connection between project activities and the outcome. Outcomes typically relate to changes in behaviours, practices, or rules/norms that govern the actions of relevant stakeholders. Examples of outcomes include strengthened/new laws or national frameworks, increased private sector actors' compliance with voluntary standards or codes of conduct, or professional applying anti-corruption knowledge and skills gained during trainings in their work. 
c.	“IMPACT” means mid- to long-term changes that the project likely has contributed to (or may contribute to). In the context of the Siemens Integrity Initiative, project impact relates positive changes in relation to the ability of actors to conduct 'clean business' in the targeted market(s). 

"""

    # Claude API costs per 1K tokens (as of March 2024)
    CLAUDE_COSTS = {
        'claude-3-opus-20240229': {
            'input': 0.015,     # $0.015 per 1K input tokens
            'output': 0.075,    # $0.075 per 1K output tokens
            'cache': 0.00375    # $0.00375 per 1K cached tokens (25% of input cost)
        },
        'claude-3-sonnet-20240229': {
            'input': 0.003,     # $0.003 per 1K input tokens
            'output': 0.015,    # $0.015 per 1K output tokens
            'cache': 0.00075    # $0.00075 per 1K cached tokens (25% of input cost)
        },
        'claude-3-haiku-20240307': {
            'input': 0.0015,    # $0.0015 per 1K input tokens
            'output': 0.0075,   # $0.0075 per 1K output tokens
            'cache': 0.000375   # $0.000375 per 1K cached tokens (25% of input cost)
        }
    }

    ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY')
    if ENCRYPTION_KEY is None:
        raise ValueError("ENCRYPTION_KEY must be set in environment variables")