import React, { useState } from 'react';
import { Carousel } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import universityImage from '../UAB-Blazers-logo.png';

function Login() {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loggedIn, setLoggedIn] = useState(false);
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [loading, setLoading] = useState(false);
    const [greeting, setGreeting] = useState('');
    const [isSignup, setIsSignup] = useState(false);
    
    // Student information states
    const [major, setMajor] = useState('');
    const [enrollmentStatus, setEnrollmentStatus] = useState('');
    const [academicLevel, setAcademicLevel] = useState('');
    const [expectedGraduation, setExpectedGraduation] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!email.trim() || !password.trim()) {
            setError('Please enter both email and password');
            return;
        }
        
        try {
            const response = await fetch('http://localhost:5000/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password }),
                credentials: 'include'
            });

            const data = await response.json();

            if (response.ok) {
                setLoggedIn(true);
                setGreeting(data.user.greeting || `Welcome back, ${data.user.name}!`);
                setError('');
            } else {
                setError(data.error || 'Error logging in');
            }
        } catch (err) {
            setError('Error connecting to server. Please make sure the backend server is running.');
        }
    };

    const handleSignup = async (e) => {
        e.preventDefault();
        if (!name.trim() || !email.trim() || !password.trim() || !major.trim() || !enrollmentStatus || !academicLevel || !expectedGraduation) {
            setError('Please fill in all required fields');
            return;
        }

        // Validate UAB email format
        if (!email.endsWith('@uab.edu')) {
            setError('Please use your UAB email address');
            return;
        }

        try {
            const response = await fetch('http://localhost:5000/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    name,
                    email,
                    password,
                    major,
                    enrollmentStatus,
                    academicLevel,
                    expectedGraduation
                }),
                credentials: 'include'
            });

            const data = await response.json();

            if (response.ok) {
                setLoggedIn(true);
                setGreeting(data.user.greeting || `Welcome, ${data.user.name}!`);
                setError('');
            } else {
                setError(data.error || 'Error signing up');
            }
        } catch (err) {
            setError('Error connecting to server. Please make sure the backend server is running.');
        }
    };

    const askQuestion = async () => {
        if (!question.trim()) return;
        
        setLoading(true);
        try {
            const response = await fetch('http://localhost:5000/ask_question', {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });
            
            const data = await response.json();
            
            if (response.ok) {
                setAnswer(data.answer);
                setError('');
                setQuestion(''); // Clear the question input after successful response
            } else {
                if (response.status === 401) {
                    setError('Session expired. Please log in again.');
                    setLoggedIn(false);
                } else {
                    setError(data.error || 'Error processing question');
                }
                setAnswer('');
            }
        } catch (err) {
            setError('Error connecting to server. Please try again.');
            setAnswer('');
        } finally {
            setLoading(false);
        }
    };

    const handleLogout = async () => {
        try {
            const response = await fetch('http://localhost:5000/logout', {
                method: 'POST',
                credentials: 'include',
            });
            if (response.ok) {
                setLoggedIn(false);
                setName('');
                setQuestion('');
                setAnswer('');
                setGreeting('');
                setError('');
            }
        } catch (err) {
            setError('Logout failed');
        }
    };

    if (loggedIn) {
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
                <h1 style={{color:"darkgreen"}}>UAB Student AI Assistant</h1>
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
                <div style={{ 
                    display: 'flex', 
                    justifyContent: 'center', 
                    gap: '10px',
                    marginBottom: '20px'
                }}>
                    <input
                        type="text"
                        placeholder="Ask a question about fees"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && askQuestion()}
                        style={{ 
                            padding: '10px', 
                            width: '300px',
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
                            cursor: 'pointer',
                            fontWeight: 'bold'
                        }}
                    >
                        {loading ? 'Loading...' : 'Ask'}
                    </button>
                </div>
                
                {error && <p style={{ color: 'red' }}>{error}</p>}
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
                        <p>{answer}</p>
                    </div>
                )}
            </div>
        );
    }

    return (
        <div>
            {/* Hero Section */}
            <section className="hero text-center text-white" style={{ 
                background: 'linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d)',
                padding: '100px 20px',
                position: 'relative'
            }}>
                <img
                    src={universityImage}
                    alt="UAB Logo"
                    style={{ width: '100px', position: 'absolute', top: '20px', left: '20px' }}
                />
                <h1>UAB AI Chat Assistant</h1>
                <h3>Simplify Your UAB Life</h3>
                <p>Ask anything, anytime—fees, courses, or library hours!</p>
                <a href="/dashboard" className="btn btn-success btn-lg">Get Started</a>
            </section>

            {/* Problem Section */}
            <section className="container my-5">
                <div className="row">
                    <div className="col-md-8">
                        <h2 className="text-success">University Websites Are a Maze</h2>
                        <p>
                            Struggling to find fees or course info? Websites bury details, waste time, and stress you out—especially on your phone!
                        </p>
                        <ul>
                            <li>Too many clicks for simple answers.</li>
                            <li>Slow responses across time zones.</li>
                            <li>Hard to navigate when busy or new.</li>
                        </ul>
                    </div>
                    <div className="col-md-4">
                        <img
                            src="/website-screenshot.jpg"
                            alt="Cluttered Website"
                            className="img-fluid"
                        />
                    </div>
                </div>
            </section>

            {/* Solution Section */}
            <section className="bg-light py-5">
                <div className="container">
                    <h2 className="text-center">Instant Answers, Just for You</h2>
                    <p className="text-center">
                        Our AI Chat Assistant is your 24/7 guide to UAB life—ask naturally, get clear answers!
                    </p>
                    <div className="row text-center">
                        <div className="col-md-3">
                            <h4>Personalization</h4>
                            <p>Log in for tailored replies, like "Sarah, your $2,000 is due May 15th!"</p>
                        </div>
                        <div className="col-md-3">
                            <h4>Step-by-Step Guidance</h4>
                            <p>Clear steps for registration: "1. Log into uab.edu…"</p>
                        </div>
                        <div className="col-md-3">
                            <h4>Quick FAQs</h4>
                            <p>Instant answers: "Wi-Fi password? Blazer2025!"</p>
                        </div>
                        <div className="col-md-3">
                            <h4>Resources</h4>
                            <p>Find help: "Tutoring at the Learning Center, 10 AM-4 PM."</p>
                        </div>
                    </div>
                    <div className="text-center mt-4">
                        <img
                            src="/chat-mockup.jpg"
                            alt="Chat Mockup"
                            style={{ width: '50%' }}
                        />
                    </div>
                </div>
            </section>

            {/* Examples Section */}
            <section className="container my-5">
                <h2 className="text-center text-success">Real Help, Real Fast</h2>
                <p className="text-center">Here's how it works for UAB students:</p>
                <Carousel>
                    <Carousel.Item>
                        <div className="chat-bubble">
                            Hey Sarah, your Intro to AI class starts next week!
                        </div>
                    </Carousel.Item>
                    <Carousel.Item>
                        <div className="chat-bubble">
                            How to register? 1. Log into uab.edu…
                        </div>
                    </Carousel.Item>
                    <Carousel.Item>
                        <div className="chat-bubble">
                            Wi-Fi password? It's 'Blazer2025'!
                        </div>
                    </Carousel.Item>
                    <Carousel.Item>
                        <div className="chat-bubble">
                            Math 101 help? Tutoring at 10 AM-4 PM.
                        </div>
                    </Carousel.Item>
                </Carousel>
                <div className="text-center mt-4">
                    <a href="/dashboard" className="btn btn-success">Try It Now</a>
                </div>
            </section>

            {/* Benefits Section */}
            <section className="bg-success text-white py-5">
                <div className="container">
                    <div className="row">
                        <div className="col-md-8">
                            <h2>A Huge Win for UAB Students</h2>
                            <p>Save time, ditch stress, and get answers anywhere—easier UAB life!</p>
                            <div className="row">
                                <div className="col-md-6">
                                    <p><strong>Saves Time</strong>: Seconds, not minutes.</p>
                                    <p><strong>Reduces Stress</strong>: No "Where do I look?"</p>
                                </div>
                                <div className="col-md-6">
                                    <p><strong>Easy to Use</strong>: Like texting a friend.</p>
                                    <p><strong>Boosts Success</strong>: Connects to tutoring, aid.</p>
                                </div>
                            </div>
                        </div>
                        <div className="col-md-4">
                            <img
                                src="/happy-student.jpg"
                                alt="Happy Student"
                                className="img-fluid"
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* Login/Signup Form Section */}
            <section id="login" className="container my-5">
                <div className="row justify-content-center">
                    <div className="col-md-6">
                        <div style={{ 
                            padding: '20px',
                            backgroundColor: '#f8f9fa',
                            borderRadius: '8px',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                        }}>
                            <div style={{ 
                                display: 'flex', 
                                justifyContent: 'center', 
                                gap: '20px',
                                marginBottom: '20px'
                            }}>
                                <button
                                    onClick={() => setIsSignup(false)}
                                    style={{
                                        padding: '10px 20px',
                                        backgroundColor: !isSignup ? '#4CAF50' : '#6c757d',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '4px',
                                        cursor: 'pointer',
                                        fontWeight: 'bold'
                                    }}
                                >
                                    Login
                                </button>
                                <button
                                    onClick={() => setIsSignup(true)}
                                    style={{
                                        padding: '10px 20px',
                                        backgroundColor: isSignup ? '#4CAF50' : '#6c757d',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '4px',
                                        cursor: 'pointer',
                                        fontWeight: 'bold'
                                    }}
                                >
                                    Sign Up
                                </button>
                            </div>

                            <h2 style={{ 
                                textAlign: 'center', 
                                color: '#2c3e50',
                                marginBottom: '20px'
                            }}>
                                {isSignup ? 'UAB Student Registration' : 'Welcome Back'}
                            </h2>
                            
                            {error && <p style={{ color: 'red', textAlign: 'center' }}>{error}</p>}
                            
                            <form onSubmit={isSignup ? handleSignup : handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                                {isSignup && (
                                    <input
                                        type="text"
                                        name="name"
                                        placeholder="Enter your full name"
                                        value={name}
                                        onChange={(e) => setName(e.target.value)}
                                        required
                                        style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
                                    />
                                )}
                                
                                <input
                                    type="email"
                                    name="email"
                                    placeholder="Enter your UAB email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    required
                                    style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
                                />
                                
                                <input
                                    type="password"
                                    name="password"
                                    placeholder="Enter your password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                    style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
                                />
                                
                                {isSignup && (
                                    <>
                                        <select
                                            name="major"
                                            value={major}
                                            onChange={(e) => setMajor(e.target.value)}
                                            required
                                            style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
                                        >
                                            <option value="">Select your major</option>
                                            <option value="Computer Science">Computer Science</option>
                                            <option value="Engineering">Engineering</option>
                                            <option value="Business">Business</option>
                                            <option value="Medicine">Medicine</option>
                                            <option value="Nursing">Nursing</option>
                                            <option value="Arts and Sciences">Arts and Sciences</option>
                                            <option value="Education">Education</option>
                                            <option value="Public Health">Public Health</option>
                                            <option value="Other">Other</option>
                                        </select>
                                        <select
                                            name="enrollmentStatus"
                                            value={enrollmentStatus}
                                            onChange={(e) => setEnrollmentStatus(e.target.value)}
                                            required
                                            style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
                                        >
                                            <option value="">Select enrollment status</option>
                                            <option value="Full-time">Full-time</option>
                                            <option value="Part-time">Part-time</option>
                                            <option value="Graduate">Graduate</option>
                                        </select>
                                        <select
                                            name="academicLevel"
                                            value={academicLevel}
                                            onChange={(e) => setAcademicLevel(e.target.value)}
                                            required
                                            style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
                                        >
                                            <option value="">Select academic level</option>
                                            <option value="Freshman">Freshman</option>
                                            <option value="Sophomore">Sophomore</option>
                                            <option value="Junior">Junior</option>
                                            <option value="Senior">Senior</option>
                                            <option value="Graduate">Graduate</option>
                                        </select>
                                        <select
                                            name="expectedGraduation"
                                            value={expectedGraduation}
                                            onChange={(e) => setExpectedGraduation(e.target.value)}
                                            required
                                            style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
                                        >
                                            <option value="">Select expected graduation year</option>
                                            <option value="2024">2024</option>
                                            <option value="2025">2025</option>
                                            <option value="2026">2026</option>
                                            <option value="2027">2027</option>
                                            <option value="2028">2028</option>
                                            <option value="2029">2029</option>
                                        </select>
                                    </>
                                )}
                                
                                <button 
                                    type="submit"
                                    style={{
                                        padding: '12px',
                                        backgroundColor: '#4CAF50',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '4px',
                                        cursor: 'pointer',
                                        fontSize: '16px',
                                        fontWeight: 'bold'
                                    }}
                                >
                                    {isSignup ? 'Register' : 'Login'}
                                </button>
                            </form>
                            
                            <p style={{ textAlign: 'center', marginTop: '15px' }}>
                                {isSignup 
                                    ? 'Already have an account? Click Login above!' 
                                    : 'New to UAB? Click Sign Up to register!'}
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="bg-dark text-white text-center py-3">
                <p>© 2025 UAB AI Chat Assistant | Developed by Nikhil Bairi | Powered by Gemini-Pro</p>
                <p>
                    <a href="#" className="text-white">About</a> |{' '}
                    <a href="#" className="text-white">Privacy</a> |{' '}
                    <a href="mailto:nbairi@uab.edu" className="text-white">Contact</a>
                </p>
            </footer>
        </div>
    );
}

export default Login; 