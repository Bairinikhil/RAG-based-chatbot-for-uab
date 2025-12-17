import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import universityImage from '../UAB-Blazers-logo.png';

function FeeChecker() {
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [greeting, setGreeting] = useState('');
    const navigate = useNavigate();
    
    // Program Finder state
    const [programs, setPrograms] = useState([]);
    const [programSearch, setProgramSearch] = useState('');
    const [selectedSchool, setSelectedSchool] = useState('');
    const [selectedLevel, setSelectedLevel] = useState('');
    const [schools, setSchools] = useState([]);
    const [levels, setLevels] = useState([]);
    const [programLoading, setProgramLoading] = useState(false);
    const [showProgramFinder, setShowProgramFinder] = useState(false);

    // RAG debug: show retrieved chunks
    const [retrievalStage, setRetrievalStage] = useState('');
    const [retrievedChunks, setRetrievedChunks] = useState([]);
    const [showChunks, setShowChunks] = useState(true);

    useEffect(() => {
        // Set a default greeting since we're bypassing authentication
        setGreeting("Welcome to UAB AI Chat Assistant! How can I help you today?");
    }, []);

    const loadProgramData = async () => {
        try {
            setProgramLoading(true);
            const [programsResponse, schoolsResponse, statsResponse] = await Promise.all([
                fetch('http://localhost:5000/api/programs'),
                fetch('http://localhost:5000/api/programs/schools'),
                fetch('http://localhost:5000/api/programs/stats')
            ]);
            
            const programsData = await programsResponse.json();
            const schoolsData = await schoolsResponse.json();
            const statsData = await statsResponse.json();
            
            if (programsData.success) {
                setPrograms(programsData.programs);
            }
            if (schoolsData.success) {
                setSchools(schoolsData.schools);
            }
            if (statsData.success) {
                setLevels(statsData.stats.levels);
            }
        } catch (err) {
            console.error('Error loading program data:', err);
        } finally {
            setProgramLoading(false);
        }
    };

    const searchPrograms = async () => {
        try {
            setProgramLoading(true);
            const params = new URLSearchParams();
            if (programSearch) params.append('search', programSearch);
            if (selectedSchool) params.append('school', selectedSchool);
            if (selectedLevel) params.append('level', selectedLevel);
            
            const response = await fetch(`http://localhost:5000/api/programs?${params}`);
            const data = await response.json();
            
            if (data.success) {
                setPrograms(data.programs);
            } else {
                setError(data.error || 'Error searching programs');
            }
        } catch (err) {
            setError('Error connecting to server. Please try again.');
        } finally {
            setProgramLoading(false);
        }
    };

    const clearFilters = () => {
        setProgramSearch('');
        setSelectedSchool('');
        setSelectedLevel('');
        loadProgramData();
    };

    const askQuestion = async () => {
        if (!question.trim()) return;
        
        setLoading(true);
        try {
            const response = await fetch('http://localhost:5000/api/chat?include_chunks=1', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                setAnswer(data.answer);
                setError('');
                setQuestion(''); // Clear the question input after successful response
                // Capture retrieval debug info when present
                setRetrievalStage(data.retrieval_stage || '');
                setRetrievedChunks(Array.isArray(data.chunks) ? data.chunks : []);
            } else {
                setError(data.error || 'Error processing question');
                setAnswer('');
                setRetrievalStage('');
                setRetrievedChunks([]);
            }
        } catch (err) {
            setError('Error connecting to server. Please try again.');
            setAnswer('');
            setRetrievalStage('');
            setRetrievedChunks([]);
        } finally {
            setLoading(false);
        }
    };

    const handleLogout = () => {
        // No logout needed since we're not using authentication
        navigate('/login');
    };

    // Render helper: convert simple markdown links to clickable anchors
    const renderAnswerHtml = (text) => {
        if (!text) return { __html: '' };
        // Convert [label](url) to <a href="url">label</a>
        let html = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        // Preserve line breaks
        html = html.replace(/\n/g, '<br/>' );
        return { __html: html };
    };

    return (
        <div style={{ textAlign: 'center', padding: '20px' }}>
            <div style={{ 
                position: 'absolute',
                top: '20px',
                right: '20px'
            }}>
                <button 
                    onClick={handleLogout}
                    style={{
                        padding: '8px 16px',
                        backgroundColor: '#dc3545',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontWeight: 'bold'
                    }}
                    onMouseOver={(e) => e.target.style.backgroundColor = '#c82333'}
                    onMouseOut={(e) => e.target.style.backgroundColor = '#dc3545'}
                >
                    Logout
                </button>
            </div>
            <img src={universityImage} alt="University Logo" style={{ maxWidth: '200px', marginBottom: '20px' }} />
            <h1 style={{color:"darkgreen"}}>UAB Student Al Assistant</h1>
            <div style={{ 
                marginBottom: '20px',
                fontSize: '1.2em',
                color: '#2c3e50',
                padding: '10px',
                backgroundColor: '#f8f9fa',
                borderRadius: '5px'
            }}>
                {greeting}
            </div>
            
            {/* Program Finder temporarily removed per request */}
            <div style={{ 
                display: 'flex', 
                justifyContent: 'center', 
                gap: '10px',
                marginBottom: '20px'
            }}>
                <input
                    type="text"
                    placeholder="Ask about programs (e.g., 'What engineering programs are available?')"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && askQuestion()}
                    style={{ 
                        padding: '10px', 
                        width: '400px',
                        borderRadius: '4px',
                        border: '1px solid #ddd'
                    }}
                />
                <button 
                    onClick={askQuestion} 
                    disabled={loading} 
                    style={{ 
                        padding: '10px 20px',
                        backgroundColor: '#4CAF50',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        fontWeight: 'bold',
                        opacity: loading ? 0.7 : 1
                    }}
                >
                    {loading ? 'Loading...' : 'Ask'}
                </button>
            </div>
            
            {error && (
                <div style={{ 
                    color: 'red', 
                    backgroundColor: '#ffebee',
                    padding: '10px',
                    borderRadius: '4px',
                    margin: '10px auto',
                    maxWidth: '600px'
                }}>
                    {error}
                </div>
            )}

            {answer && (
                <div style={{
                    marginTop: '20px',
                    padding: '15px',
                    backgroundColor: '#f8f9fa',
                    borderRadius: '5px',
                    maxWidth: '600px',
                    margin: '20px auto',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                    <h4>Answer:</h4>
                    <div
                        style={{
                            whiteSpace: 'normal',
                            textAlign: 'left',
                            lineHeight: 1.6
                        }}
                        dangerouslySetInnerHTML={renderAnswerHtml(answer)}
                    />
                </div>
            )}

            {retrievedChunks.length > 0 && (
                <div style={{
                    marginTop: '10px',
                    padding: '12px',
                    backgroundColor: '#eef6ee',
                    borderRadius: '5px',
                    maxWidth: '600px',
                    margin: '10px auto',
                    textAlign: 'left'
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <strong>Retrieved Chunks {retrievalStage ? `(stage: ${retrievalStage})` : ''}</strong>
                        <button
                            onClick={() => setShowChunks(!showChunks)}
                            style={{
                                border: 'none',
                                background: 'transparent',
                                color: '#2c7a2c',
                                cursor: 'pointer',
                                fontWeight: 'bold'
                            }}
                        >
                            {showChunks ? 'Hide' : 'Show'}
                        </button>
                    </div>
                    {showChunks && (
                        <ol style={{ marginTop: '8px', paddingLeft: '20px' }}>
                            {retrievedChunks.map((c, idx) => (
                                <li key={idx} style={{ marginBottom: '8px' }}>
                                    <div style={{ fontSize: '0.9em', color: '#2c3e50' }}>
                                        [{c.info_type}] {c.text}
                                    </div>
                                    <div style={{ fontSize: '0.8em', color: '#6c757d' }}>
                                        {c.program_abbreviation ? `program: ${c.program_abbreviation}` : ''}
                                        {c.department_name ? `${c.program_abbreviation ? ' • ' : ''}dept: ${c.department_name}` : ''}
                                    </div>
                                </li>
                            ))}
                        </ol>
                    )}
                </div>
            )}
        </div>
    );
}

export default FeeChecker; 