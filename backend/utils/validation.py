"""
Validation utilities for UAB Chat Bot
Input validation and sanitization
"""

import re
from typing import Dict, List, Tuple, Optional

class ValidationHelper:
    """Helper class for input validation"""
    
    @staticmethod
    def validate_name(name: str) -> Tuple[bool, str]:
        """
        Validate user name
        
        Args:
            name: User name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name or not name.strip():
            return False, "Name is required"
        
        name = name.strip()
        if len(name) < 2:
            return False, "Name must be at least 2 characters long"
        
        if len(name) > 100:
            return False, "Name must be less than 100 characters"
        
        # Check for valid characters (letters, spaces, hyphens, apostrophes)
        if not re.match(r"^[a-zA-Z\s\-']+$", name):
            return False, "Name can only contain letters, spaces, hyphens, and apostrophes"
        
        return True, ""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """
        Validate email address
        
        Args:
            email: Email to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not email:
            return True, ""  # Email is optional
        
        email = email.strip()
        if len(email) > 254:
            return False, "Email must be less than 254 characters"
        
        # Basic email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return False, "Invalid email format"
        
        return True, ""
    
    @staticmethod
    def validate_uab_email(email: str) -> Tuple[bool, str]:
        """
        Validate UAB email address
        
        Args:
            email: Email to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not email:
            return True, ""  # Email is optional
        
        email = email.strip().lower()
        if not email.endswith('@uab.edu'):
            return False, "Please use your UAB email address"
        
        return True, ""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """
        Validate password
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not password:
            return False, "Password is required"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        
        if len(password) > 100:
            return False, "Password must be less than 100 characters"
        
        return True, ""
    
    @staticmethod
    def validate_question(question: str) -> Tuple[bool, str]:
        """
        Validate user question
        
        Args:
            question: Question to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not question or not question.strip():
            return False, "Question is required"
        
        question = question.strip()
        if len(question) < 3:
            return False, "Question must be at least 3 characters long"
        
        if len(question) > 1000:
            return False, "Question must be less than 1000 characters"
        
        return True, ""
    
    @staticmethod
    def validate_signup_data(data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate signup form data
        
        Args:
            data: Signup form data
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate name
        name = data.get('name', '')
        name_valid, name_error = ValidationHelper.validate_name(name)
        if not name_valid:
            errors.append(f"Name: {name_error}")
        
        # Validate email (optional)
        email = data.get('email', '')
        if email:
            email_valid, email_error = ValidationHelper.validate_uab_email(email)
            if not email_valid:
                errors.append(f"Email: {email_error}")
        
        # Validate required fields
        required_fields = ['major', 'enrollmentStatus', 'academicLevel', 'expectedGraduation']
        for field in required_fields:
            if not data.get(field):
                errors.append(f"{field.replace('_', ' ').title()}: This field is required")
        
        # Validate expected graduation year
        expected_graduation = data.get('expectedGraduation')
        if expected_graduation:
            try:
                year = int(expected_graduation)
                current_year = 2024
                if year < current_year or year > current_year + 10:
                    errors.append("Expected graduation: Please enter a valid year")
            except ValueError:
                errors.append("Expected graduation: Please enter a valid year")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """
        Sanitize user input
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove potentially dangerous characters
        text = text.strip()
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Limit length
        text = text[:1000]
        
        return text
