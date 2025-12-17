"""
Response helper utilities for UAB Chat Bot
Common response formatting and error handling
"""

from flask import jsonify
from typing import Dict, Any, Tuple

class ResponseHelper:
    """Helper class for formatting API responses"""
    
    @staticmethod
    def success_response(data: Dict[str, Any] = None, message: str = "Success") -> Tuple[Dict, int]:
        """
        Create a success response
        
        Args:
            data: Response data
            message: Success message
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        response = {"message": message}
        if data:
            response.update(data)
        return response, 200
    
    @staticmethod
    def error_response(message: str, status_code: int = 400, error_code: str = None) -> Tuple[Dict, int]:
        """
        Create an error response
        
        Args:
            message: Error message
            status_code: HTTP status code
            error_code: Optional error code
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        response = {"error": message}
        if error_code:
            response["error_code"] = error_code
        return response, status_code
    
    @staticmethod
    def validation_error_response(message: str, field: str = None) -> Tuple[Dict, int]:
        """
        Create a validation error response
        
        Args:
            message: Validation error message
            field: Field that failed validation
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        response = {"error": message}
        if field:
            response["field"] = field
        return response, 400
    
    @staticmethod
    def authentication_error_response(message: str = "Authentication required") -> Tuple[Dict, int]:
        """
        Create an authentication error response
        
        Args:
            message: Authentication error message
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        return {"error": message}, 401
    
    @staticmethod
    def not_found_response(message: str = "Resource not found") -> Tuple[Dict, int]:
        """
        Create a not found response
        
        Args:
            message: Not found message
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        return {"error": message}, 404
    
    @staticmethod
    def server_error_response(message: str = "Internal server error") -> Tuple[Dict, int]:
        """
        Create a server error response
        
        Args:
            message: Server error message
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        return {"error": message}, 500
    
    @staticmethod
    def handle_service_response(service_success: bool, service_data: Any, service_message: str) -> Tuple[Dict, int]:
        """
        Handle response from service layer
        
        Args:
            service_success: Service operation success
            service_data: Service response data
            service_message: Service message
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        if service_success:
            return ResponseHelper.success_response(service_data, service_message)
        else:
            return ResponseHelper.error_response(service_message, 500)



