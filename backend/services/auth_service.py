"""
Authentication service for UAB Chat Bot
Handles user authentication, session management, and authorization
"""

from flask import session
from database_models import Student, User
from typing import Dict, Optional, Tuple

class AuthService:
    """Service for handling authentication operations"""
    
    @staticmethod
    def login_user(email: str, password: str) -> Tuple[bool, Dict, str]:
        """
        Authenticate a user by email and password
        
        Args:
            email: User's email
            password: User's password
            
        Returns:
            Tuple of (success, user_data, message)
        """
        try:
            # Check if user exists by email
            user = User.get_user_by_email(email)
            
            if user:
                # Verify password (in a real app, you'd hash and compare)
                # For now, we'll accept any password for existing users
                session['user_id'] = user['id']
                session['user_email'] = email
                
                # Get student data if available
                student = Student.get_student_by_name(user['name']) if user['name'] else None
                
                return True, {
                    "message": "Login successful",
                    "name": user['name'],
                    "email": user['email'],
                    "isStudent": student is not None,
                    "student_data": Student.to_dict(student) if student else None
                }, "Login successful"
            else:
                # For demo purposes, create a new user with the email as name
                # In a real app, you'd require proper registration first
                name = email.split('@')[0]  # Use email prefix as name
                
                # Create new user
                new_user_data = {
                    'name': name,
                    'email': email,
                    'password': password  # In real app, this should be hashed
                }
                
                user = User.create_user(new_user_data)
                session['user_id'] = user['id']
                session['user_email'] = email
                
                # Create corresponding student record
                new_student_data = {
                    'name': name,
                    'email': email,
                    'major': 'Computer Science',  # Default values
                    'enrollment_status': 'Full-time',
                    'academic_level': 'Undergraduate',
                    'expected_graduation': 2025
                }
                
                student = Student.create_student(new_student_data)
                
                return True, {
                    "message": "Login successful",
                    "name": name,
                    "email": email,
                    "isStudent": True,
                    "student_data": Student.to_dict(student)
                }, "New user created and logged in"
                
        except Exception as e:
            return False, {"error": f"Authentication error: {str(e)}"}, str(e)
    
    @staticmethod
    def signup_user(user_data: Dict) -> Tuple[bool, Dict, str]:
        """
        Register a new user
        
        Args:
            user_data: Dictionary containing user information
            
        Returns:
            Tuple of (success, user_data, message)
        """
        try:
            name = user_data.get('name')
            email = user_data.get('email')
            major = user_data.get('major')
            enrollment_status = user_data.get('enrollmentStatus')
            academic_level = user_data.get('academicLevel')
            expected_graduation = user_data.get('expectedGraduation')

            if not all([name, major, enrollment_status, academic_level, expected_graduation]):
                return False, {"error": "Required fields are missing"}, "Missing required fields"

            # Create new student
            new_student_data = {
                'name': name,
                'email': email,
                'major': major,
                'enrollment_status': enrollment_status,
                'academic_level': academic_level,
                'expected_graduation': expected_graduation
            }
            
            student = Student.create_student(new_student_data)
            session['student_name'] = name
            
            first_name = name.split()[0]
            greeting = f"Hi {first_name}! Welcome to UAB AI Chat Assistant. How can I help you today?"
            
            return True, {
                "message": "Signup successful",
                "name": name,
                "greeting": greeting,
                "isStudent": True,
                "student_data": Student.to_dict(student)
            }, "User registered successfully"
            
        except Exception as e:
            return False, {"error": f"Registration error: {str(e)}"}, str(e)
    
    @staticmethod
    def logout_user() -> Tuple[bool, Dict, str]:
        """
        Logout the current user
        
        Returns:
            Tuple of (success, response_data, message)
        """
        try:
            session.clear()
            return True, {"message": "Logged out successfully"}, "Logout successful"
        except Exception as e:
            return False, {"error": f"Logout error: {str(e)}"}, str(e)
    
    @staticmethod
    def get_current_user() -> Tuple[bool, Dict, str]:
        """
        Get current authenticated user
        
        Returns:
            Tuple of (success, user_data, message)
        """
        try:
            if 'user_id' not in session or 'user_email' not in session:
                return False, {"error": "Not logged in"}, "No active session"
            
            user_id = session['user_id']
            user_email = session['user_email']
            
            # Get user from database
            user = User.get_user_by_id(user_id)
            if not user:
                session.clear()
                return False, {"error": "User not found"}, "Invalid session"
            
            # Get student data if available
            student = Student.get_student_by_name(user['name']) if user['name'] else None
            print(f"DEBUG: user['name'] = {user['name']}")
            print(f"DEBUG: student = {student}")
                
            return True, {
                "message": "Session valid",
                "name": user['name'],
                "email": user['email'],
                "student_data": Student.to_dict(student) if student else None
            }, "User session valid"
            
        except Exception as e:
            return False, {"error": f"Session error: {str(e)}"}, str(e)
    
    @staticmethod
    def is_authenticated() -> bool:
        """
        Check if user is currently authenticated
        
        Returns:
            Boolean indicating authentication status
        """
        return 'user_id' in session and 'user_email' in session
