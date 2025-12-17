#!/usr/bin/env python3
"""
Improved UAB Chat Bot Backend
Using enhanced RAG service with better knowledge base
"""

from flask import Flask, request, jsonify, session
import re
import logging
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file FIRST
load_dotenv()

from config import get_config
from database_models import get_db_connection
from services.auth_service import AuthService
from services.database_service import DatabaseService
from services.chat_service import ChatService
from program_loader import ProgramLoader
from services.ai_service import AIService as RealAIService
from utils.response_helpers import ResponseHelper
from utils.validation import ValidationHelper

# Import new services
from services.entity_extractor import EntityExtractor, set_entity_extractor
from services.entity_cache import init_entity_cache
from services.rate_limiter import init_gemini_rate_limiter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config = get_config()
app.config.update({
    'SECRET_KEY': config.SECRET_KEY,
    'SESSION_COOKIE_SECURE': config.SESSION_COOKIE_SECURE,
    'SESSION_COOKIE_SAMESITE': config.SESSION_COOKIE_SAMESITE
})

# Initialize CORS
CORS(app, 
     origins=config.ALLOWED_ORIGINS,
     supports_credentials=True,
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

# Initialize services
auth_service = AuthService()
database_service = DatabaseService()
program_loader = ProgramLoader()  # Load program index into memory
chat_service = ChatService(program_loader)  # Initialize chat service
ai_service = None

# Routes
@app.route('/')
def home():
    return jsonify({"message": "UAB Chat Bot Backend - Improved Version", "status": "running"})

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Check database connection
        conn = get_db_connection()
        conn.close()
        
        # Check AI service
        ai_status = "available" if (ai_service and ai_service.is_available()) else "unavailable"
        
        # Check program loader
        program_status = "available" if program_loader.is_loaded() else "unavailable"
        program_count = len(program_loader.get_all_programs()) if program_loader.is_loaded() else 0
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "ai_service": ai_status,
            "program_service": program_status,
            "programs_loaded": program_count,
            "version": "improved_v2"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/signup', methods=['POST', 'OPTIONS'])
def signup():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        force_rag = request.args.get('force_rag', '').lower() in ('1', 'true', 'yes')
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        major = data.get('major', '').strip()
        academic_level = data.get('academic_level', '').strip()
        
        # Validate input
        name_valid, name_error = ValidationHelper.validate_name(name)
        email_valid, email_error = ValidationHelper.validate_email(email)
        password_valid, password_error = ValidationHelper.validate_password(password)
        
        if not all([name_valid, email_valid, password_valid]):
            errors = []
            if not name_valid: errors.append(name_error)
            if not email_valid: errors.append(email_error)
            if not password_valid: errors.append(password_error)
            return jsonify(ResponseHelper.validation_error_response("; ".join(errors)))
        
        # Attempt signup
        success, user_data, message = auth_service.signup_user(
            name, email, password, major, academic_level
        )
        
        if success:
            return jsonify(ResponseHelper.success_response({
                "message": "Account created successfully",
                "user": user_data
            }))
        else:
            return jsonify(ResponseHelper.validation_error_response(message))
    
    except Exception as e:
        return jsonify(ResponseHelper.server_error_response(f"Signup failed: {str(e)}"))

@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        
        # Validate input
        email_valid, email_error = ValidationHelper.validate_email(email)
        password_valid, password_error = ValidationHelper.validate_password(password)
        
        if not email_valid:
            return jsonify(ResponseHelper.validation_error_response(email_error))
        if not password_valid:
            return jsonify(ResponseHelper.validation_error_response(password_error))
        
        # Attempt login
        success, user_data, message = auth_service.login_user(email, password)
        
        if success:
            return jsonify(ResponseHelper.success_response({
                "message": "Login successful",
                "user": user_data
            }))
        else:
            return jsonify(ResponseHelper.authentication_error_response(message))
    
    except Exception as e:
        return jsonify(ResponseHelper.server_error_response(f"Login failed: {str(e)}"))

@app.route('/ask_question', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        question = data.get('question') if data else None
        
        # Validate input
        question_valid, question_error = ValidationHelper.validate_question(question or '')
        if not question_valid:
            response, status = ResponseHelper.validation_error_response(question_error)
            return jsonify(response), status
        
        # Generate AI response using improved RAG (no user context for now)
        ai_success, ai_response, ai_error = ai_service.generate_response(
            question, 
            None  # No student context since we're not using authentication
        )
        
        if ai_success:
            response, status = ResponseHelper.success_response({"answer": ai_response})
            return jsonify(response), status
        else:
            response, status = ResponseHelper.server_error_response(ai_error)
            return jsonify(response), status
    
    except Exception as e:
        response, status = ResponseHelper.server_error_response(f"Question processing failed: {str(e)}")
        return jsonify(response), status

@app.route('/logout', methods=['POST', 'OPTIONS'])
def logout():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        success, message = auth_service.logout_user()
        if success:
            return jsonify(ResponseHelper.success_response({"message": "Logged out successfully"}))
        else:
            return jsonify(ResponseHelper.server_error_response(message))
    except Exception as e:
        return jsonify(ResponseHelper.server_error_response(f"Logout failed: {str(e)}"))

@app.route('/check_session', methods=['GET'])
def check_session():
    try:
        if auth_service.is_authenticated():
            success, user_data, message = auth_service.get_current_user()
            if success:
                return jsonify(ResponseHelper.success_response({"user": user_data}))
        
        return jsonify(ResponseHelper.authentication_error_response())
    except Exception as e:
        return jsonify(ResponseHelper.server_error_response(f"Session check failed: {str(e)}"))

@app.route('/degrees/<degree_level>', methods=['GET'])
def get_degrees_by_level(degree_level):
    try:
        if not auth_service.is_authenticated():
            return jsonify(ResponseHelper.authentication_error_response())
        
        degrees = database_service.get_degrees_by_level(degree_level)
        return jsonify(ResponseHelper.success_response({"degrees": degrees}))
    except Exception as e:
        return jsonify(ResponseHelper.server_error_response(f"Failed to get degrees: {str(e)}"))

@app.route('/fees/<fee_type>', methods=['GET'])
def get_fees_by_type(fee_type):
    try:
        if not auth_service.is_authenticated():
            return jsonify(ResponseHelper.authentication_error_response())
        
        fees = database_service.get_fees_by_type(fee_type)
        return jsonify(ResponseHelper.success_response({"fees": fees}))
    except Exception as e:
        return jsonify(ResponseHelper.server_error_response(f"Failed to get fees: {str(e)}"))

@app.route('/init-db', methods=['POST'])
def init_database():
    try:
        success, message = database_service.init_database()
        if success:
            return jsonify(ResponseHelper.success_response({"message": "Database initialized successfully"}))
        else:
            return jsonify(ResponseHelper.server_error_response(message))
    except Exception as e:
        return jsonify(ResponseHelper.server_error_response(f"Database initialization failed: {str(e)}"))

@app.route('/knowledge-base/stats', methods=['GET'])
def get_knowledge_base_stats():
    """Get statistics about the knowledge base"""
    try:
        if not auth_service.is_authenticated():
            return jsonify(ResponseHelper.authentication_error_response())
        
        if ai_service and ai_service.rag_service:
            stats = ai_service.rag_service.get_knowledge_base_stats()
            return jsonify(ResponseHelper.success_response({"stats": stats}))
        else:
            return jsonify(ResponseHelper.server_error_response("AI service not available"))
    except Exception as e:
        return jsonify(ResponseHelper.server_error_response(f"Failed to get knowledge base stats: {str(e)}"))

# Program Finder API Endpoints - Simple and Fast
@app.route('/api/programs', methods=['GET'])
def get_programs():
    """Get programs with optional filtering by school, level, or search query"""
    try:
        # Get query parameters
        school = request.args.get('school', '').strip()
        level = request.args.get('level', '').strip()
        search = request.args.get('search', '').strip()
        
        # Start with all programs
        programs = program_loader.get_all_programs()
        
        # Apply filters
        if school:
            programs = [p for p in programs if p.get('school', '').lower() == school.lower()]
        
        if level:
            programs = [p for p in programs if p.get('level', '').lower() == level.lower()]
        
        if search:
            search_lower = search.lower()
            programs = [p for p in programs if search_lower in p.get('program_name', '').lower()]
        
        return jsonify({
            "success": True,
            "programs": programs,
            "total": len(programs),
            "filters": {
                "school": school,
                "level": level,
                "search": search
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to get programs: {str(e)}"
        }), 500

@app.route('/api/programs/schools', methods=['GET'])
def get_schools():
    """Get list of all schools"""
    try:
        schools = program_loader.get_schools()
        return jsonify({
            "success": True,
            "schools": schools,
            "total": len(schools)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to get schools: {str(e)}"
        }), 500

@app.route('/api/programs/stats', methods=['GET'])
def get_program_stats():
    """Get program statistics"""
    try:
        stats = program_loader.get_stats()
        return jsonify({
            "success": True,
            "stats": stats
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to get program stats: {str(e)}"
        }), 500

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Handle natural language chat queries about programs"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip() if data else ''
        
        if not question:
            return jsonify({
                "success": False,
                "error": "Please provide a question"
            }), 400
        
        # Use AI RAG service when available
        if ai_service and ai_service.is_available():
            include_chunks = request.args.get('include_chunks', '').lower() in ('1','true','yes')

            # Extract entities ONCE at the entry point
            extracted_entities = None
            if getattr(ai_service, 'rag_service', None):
                try:
                    rag = ai_service.rag_service
                    # Extract entities once and pass to both generate and retrieve
                    dept, prog, info = rag._extract_entities(question, use_cache=True)
                    extracted_entities = (dept, prog, info)
                    logger.debug(f"Extracted entities once: dept={dept}, prog={prog}, info={info}")
                except Exception as e:
                    logger.warning(f"Entity extraction failed: {e}")

            # Generate response with pre-extracted entities
            # Use Enhanced RAG V2 to get response AND metadata including chunks
            if hasattr(ai_service, 'rag_service') and hasattr(ai_service.rag_service, 'generate_enhanced_response_v2'):
                response, metadata = ai_service.rag_service.generate_enhanced_response_v2(
                    question,
                    student_context=None,
                    extracted_entities=extracted_entities,
                    use_hybrid=False,
                    use_reranking=True
                )

                if response:
                    payload = {
                        "success": True,
                        "answer": response,
                        "type": "rag",
                        "retrieval_stage": metadata.get("retrieval_stage")
                    }

                    # Include chunks from metadata if requested
                    if include_chunks and "chunks_data" in metadata:
                        payload["chunks"] = [
                            {
                                "info_type": c.get("info_type"),
                                "program_abbreviation": c.get("program_abbreviation"),
                                "department_name": c.get("department_name"),
                                "text": c.get("chunk_text")
                            } for c in metadata["chunks_data"]
                        ]

                    logger.info(f"Enhanced RAG V2 response: stage={metadata.get('retrieval_stage')}, chunks={len(metadata.get('chunks_data', []))}")
                    return jsonify(payload)
                else:
                    return jsonify({
                        "success": False,
                        "error": "RAG failed to generate a response"
                    }), 500
            else:
                # Fallback to old generate_response method
                ai_success, ai_response, ai_error = ai_service.generate_response(
                    question, None, extracted_entities=extracted_entities
                )

                if ai_success:
                    payload = {
                        "success": True,
                        "answer": ai_response,
                        "type": "rag"
                    }
                    return jsonify(payload)
            return jsonify({
                "success": False,
                "error": ai_error or "RAG failed to generate a response"
            }), 500
        
        # Fallback to legacy program search only if AI is unavailable
        success, response, error = chat_service.process_question(question)
        if success:
            return jsonify({
                "success": True,
                "answer": response,
                "type": "program_search"
            })
        return jsonify({
            "success": False,
            "error": error
        }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to process question: {str(e)}"
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting UAB Chat Bot Backend - Improved Version")
    logger.info("📚 Using Enhanced RAG Service V2 with Entity Extraction")
    logger.info("🔧 Configuration loaded from environment")

    # Validate configuration
    try:
        config.validate()
        logger.info("✅ Configuration validated")
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        exit(1)

    # Initialize rate limiter for Gemini API (free tier: 2 requests/minute)
    try:
        logger.info("Initializing rate limiter...")
        rate_limiter = init_gemini_rate_limiter(
            requests_per_minute=2,  # Gemini free tier limit
            burst_size=3,           # Allow small bursts
            max_retries=3
        )
        logger.info("✅ Rate limiter initialized")
    except Exception as e:
        logger.warning(f"⚠️ Rate limiter initialization failed: {e}")

    # Initialize entity cache
    try:
        logger.info("Initializing entity cache...")
        entity_cache = init_entity_cache(
            max_size=1000,      # Cache up to 1000 queries
            ttl_seconds=3600    # 1 hour TTL
        )
        logger.info("✅ Entity cache initialized")
    except Exception as e:
        logger.warning(f"⚠️ Entity cache initialization failed: {e}")

    # Initialize entity extractor
    try:
        logger.info("Initializing entity extractor...")
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-exp")
            # Ensure model name has the proper prefix
            if not model_name.startswith(('models/', 'tunedModels/')):
                model_name = f"models/{model_name}"
            model = genai.GenerativeModel(model_name)
            entity_extractor = EntityExtractor(genai_model=model)
            set_entity_extractor(entity_extractor)
            logger.info("✅ Entity extractor initialized with Gemini model")
        else:
            entity_extractor = EntityExtractor(genai_model=None)
            set_entity_extractor(entity_extractor)
            logger.warning("⚠️ Entity extractor initialized without LLM (heuristics only)")
    except Exception as e:
        logger.error(f"❌ Entity extractor initialization failed: {e}")
        # Create basic extractor without LLM
        entity_extractor = EntityExtractor(genai_model=None)
        set_entity_extractor(entity_extractor)

    # Initialize AI service (pgvector if configured)
    try:
        logger.info("Initializing AI service...")
        ai_service = RealAIService()
        if ai_service.is_available():
            logger.info("✅ AI Service initialized and ready")
        else:
            logger.warning("⚠️ AI Service initialized but not fully available")
    except Exception as e:
        logger.error(f"❌ Failed to init AI service: {e}", exc_info=True)

    # Start the server
    # Note: use_reloader=False prevents issues with sentence-transformers model loading on reload
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )
