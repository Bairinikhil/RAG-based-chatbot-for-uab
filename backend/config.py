import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
try:
    # Load from the backend directory where this config.py is located
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except:
    pass  # Continue without .env file

class Config:
    """Application configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///students.db')
    CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma_db')
    
    # AI Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # CORS Configuration
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
    
    # Data Paths - CSV loading removed
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        # GEMINI_API_KEY is optional for program finder functionality
        if not cls.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY not set - AI features will be limited")
        return True

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_SAMESITE = 'Lax'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_SAMESITE = 'None'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    
    if env == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig()