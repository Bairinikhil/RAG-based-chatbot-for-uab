# RAG-based UAB Chatbot

An intelligent AI-powered chatbot for University of Alabama at Birmingham (UAB) students and prospective students. This chatbot uses RAG (Retrieval-Augmented Generation) to provide accurate information about tuition fees, application deadlines, program details, and general university information.

## Features

- **AI-Powered Q&A**: Ask questions about UAB programs, fees, deadlines, and requirements
- **RAG Architecture**: Uses vector embeddings and semantic search for accurate information retrieval
- **Real-time Responses**: Get instant answers to your queries
- **User Authentication**: Secure login system for personalized experience
- **PostgreSQL Vector Database**: Efficient storage and retrieval using pgvector
- **Hybrid Search**: Combines semantic search with keyword matching (BM25)
- **Template-based Responses**: Fast, accurate responses without expensive LLM API calls

## Tech Stack

### Frontend
- **React.js** - Modern UI library
- **React Router** - Client-side routing
- **React Bootstrap** - Responsive UI components
- **Axios** - HTTP client for API calls

### Backend
- **Python Flask** - Lightweight web framework
- **PostgreSQL** - Relational database with vector extension
- **pgvector** - Vector similarity search
- **Sentence Transformers** - Local embedding generation (no API costs!)
- **SQLAlchemy** - ORM for database operations
- **Flask-CORS** - Cross-origin resource sharing
- **BM25** - Keyword-based search ranking
- **Ollama** (Optional) - Local LLM support

## Architecture

The system uses a RAG (Retrieval-Augmented Generation) approach:

1. **Knowledge Base**: UAB-related information is chunked and stored with vector embeddings
2. **Query Processing**: User questions are converted to embeddings
3. **Similarity Search**: Vector similarity search finds relevant information
4. **Hybrid Ranking**: Combines semantic and keyword-based search results
5. **Response Generation**: Template-based formatting provides structured answers

## Prerequisites

- **Node.js** (v14 or higher)
- **Python** 3.8+
- **PostgreSQL** with pgvector extension
- **pip** (Python package manager)
- **npm** or **yarn**

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/RAG-based-UAB-chatbot.git
cd RAG-based-UAB-chatbot
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
cd backend
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

#### Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual credentials
# Required variables:
# - DATABASE_URL: Your PostgreSQL connection string
# - FIRECRAWL_API_KEY: Your Firecrawl API key (if using web scraping)
```

#### Setup Database

Make sure PostgreSQL is running with the pgvector extension installed. The application will create tables automatically on first run.

#### Run Backend Server

```bash
python app.py
```

The backend will start on `http://localhost:5000`

### 3. Frontend Setup

Open a new terminal window:

```bash
# From the project root directory
npm install

# Start the development server
npm start
```

The frontend will start on `http://localhost:3000`

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Create an account or log in
3. Start asking questions about UAB programs, fees, and deadlines!

### Example Questions

- "What is the tuition fee for the Master's in Computer Science?"
- "When is the application deadline for Fall 2026?"
- "What are the requirements for the Civil Engineering program?"
- "Tell me about financial aid options"
- "What programs are available in the School of Engineering?"

## Project Structure

```
RAG-based-UAB-chatbot/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── config.py              # Configuration settings
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example          # Environment variables template
│   ├── models/               # Database models
│   ├── services/             # Business logic and RAG system
│   ├── data/                 # Data files and knowledge base
│   └── utils/                # Helper functions
├── src/
│   ├── components/           # React components
│   │   ├── Login.js         # Authentication component
│   │   └── FeeChecker.js    # Main chatbot interface
│   ├── App.js               # Main React app
│   └── index.js             # Entry point
├── public/                   # Static assets
├── package.json             # Node dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Configuration

### Backend Configuration (.env)

- `DATABASE_URL`: PostgreSQL connection string with pgvector
- `FLASK_ENV`: Set to `development` or `production`
- `USE_SENTENCE_TRANSFORMERS`: Set to `true` for local embeddings
- `SKIP_LLM_GENERATION`: Set to `true` to use template-based responses
- `USE_OLLAMA`: Set to `true` to use local Ollama LLM (optional)

### Frontend Configuration

The frontend is pre-configured to connect to `http://localhost:5000`. For production deployment, you may need to update the API URL.

## Development

### Running Tests

```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
npm test
```

### Adding New Data

To add new information to the knowledge base:

1. Add your data to `backend/data/`
2. Use the ingestion scripts in `backend/` to process and store the data
3. Run the appropriate ingestion script (e.g., `python ingest_all_programs.py`)

## Deployment

### Backend Deployment

- Deploy the Flask app to platforms like Heroku, Railway, or AWS
- Ensure PostgreSQL with pgvector is available
- Set environment variables in your hosting platform

### Frontend Deployment

- Build the production bundle: `npm run build`
- Deploy to platforms like Vercel, Netlify, or GitHub Pages
- Update API URL to point to your production backend

## Performance

- **Local Embeddings**: No API costs for embedding generation
- **Template Responses**: Fast responses without LLM API calls
- **Vector Search**: Efficient similarity search with pgvector
- **Hybrid Search**: Better accuracy with combined semantic and keyword search

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Security

- Never commit `.env` files with real credentials
- Keep API keys secure
- Use environment variables for sensitive data
- Review `.gitignore` to ensure no sensitive files are tracked

## Troubleshooting

### Backend won't start
- Check if PostgreSQL is running
- Verify DATABASE_URL in `.env`
- Ensure all Python dependencies are installed

### Frontend can't connect to backend
- Verify backend is running on port 5000
- Check for CORS errors in browser console
- Ensure Flask-CORS is properly configured

### Database errors
- Make sure pgvector extension is installed: `CREATE EXTENSION vector;`
- Check database connection string
- Verify PostgreSQL version compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- University of Alabama at Birmingham for the data
- Sentence Transformers for embeddings
- pgvector for efficient vector search
- Flask and React communities

## Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for UAB students**
