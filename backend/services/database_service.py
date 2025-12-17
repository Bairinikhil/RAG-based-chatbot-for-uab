"""
Database service for UAB Chat Bot
Handles database operations, initialization, and maintenance
"""

from database_models import init_database, get_db_connection, Degree, Fee
from typing import Dict, List, Optional, Tuple
import os

class DatabaseService:
    """Service for handling database operations"""
    
    @staticmethod
    def initialize_database() -> Tuple[bool, str]:
        """
        Initialize the database with required tables
        
        Returns:
            Tuple of (success, message)
        """
        try:
            init_database()
            return True, "Database initialized successfully"
        except Exception as e:
            return False, f"Database initialization failed: {str(e)}"
    
    @staticmethod
    def get_database_status() -> Dict:
        """
        Get database status information
        
        Returns:
            Dictionary with database status
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                "connected": True,
                "tables": tables,
                "database_path": os.path.join(os.path.dirname(__file__), '..', 'students.db')
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "database_path": os.path.join(os.path.dirname(__file__), '..', 'students.db')
            }
    
    @staticmethod
    def get_all_degrees() -> Tuple[bool, List[Dict], str]:
        """
        Get all degrees from database
        
        Returns:
            Tuple of (success, degrees_list, error_message)
        """
        try:
            degrees = Degree.get_all_degrees()
            return True, degrees, ""
        except Exception as e:
            return False, [], f"Failed to fetch degrees: {str(e)}"
    
    @staticmethod
    def search_degrees(query: str) -> Tuple[bool, List[Dict], str]:
        """
        Search degrees by query
        
        Args:
            query: Search query
            
        Returns:
            Tuple of (success, degrees_list, error_message)
        """
        try:
            if not query or not query.strip():
                return False, [], "Search query required"
            
            degrees = Degree.search_degrees(query)
            return True, degrees, ""
        except Exception as e:
            return False, [], f"Failed to search degrees: {str(e)}"
    
    @staticmethod
    def get_degrees_by_level(level: str) -> Tuple[bool, List[Dict], str]:
        """
        Get degrees by level
        
        Args:
            level: Degree level (Bachelor, Master, PhD, etc.)
            
        Returns:
            Tuple of (success, degrees_list, error_message)
        """
        try:
            degrees = Degree.get_degrees_by_level(level)
            return True, degrees, ""
        except Exception as e:
            return False, [], f"Failed to fetch degrees by level: {str(e)}"
    
    @staticmethod
    def get_degrees_by_subject(subject: str) -> Tuple[bool, List[Dict], str]:
        """
        Get degrees by subject area
        
        Args:
            subject: Subject area
            
        Returns:
            Tuple of (success, degrees_list, error_message)
        """
        try:
            degrees = Degree.get_degrees_by_subject(subject)
            return True, degrees, ""
        except Exception as e:
            return False, [], f"Failed to fetch degrees by subject: {str(e)}"
    
    @staticmethod
    def get_degree_details(degree_id: int) -> Tuple[bool, Optional[Dict], str]:
        """
        Get degree details by ID
        
        Args:
            degree_id: Degree ID
            
        Returns:
            Tuple of (success, degree_data, error_message)
        """
        try:
            degree = Degree.get_degree_by_id(degree_id)
            if not degree:
                return False, None, "Degree not found"
            return True, degree, ""
        except Exception as e:
            return False, None, f"Failed to fetch degree details: {str(e)}"
    
    @staticmethod
    def get_student_fees(student_id: int) -> Tuple[bool, List[Dict], str]:
        """
        Get student fees
        
        Args:
            student_id: Student ID
            
        Returns:
            Tuple of (success, fees_list, error_message)
        """
        try:
            fees = Fee.get_student_fees(student_id)
            return True, fees, ""
        except Exception as e:
            return False, [], f"Failed to fetch student fees: {str(e)}"
    
    @staticmethod
    def update_fee_status(fee_id: int, status: str) -> Tuple[bool, str]:
        """
        Update fee status
        
        Args:
            fee_id: Fee ID
            status: New status
            
        Returns:
            Tuple of (success, message)
        """
        try:
            success = Fee.update_fee_status(fee_id, status)
            if success:
                return True, "Fee status updated successfully"
            else:
                return False, "Failed to update fee status"
        except Exception as e:
            return False, f"Failed to update fee status: {str(e)}"
