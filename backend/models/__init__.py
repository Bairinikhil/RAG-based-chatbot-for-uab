"""
Models package for UAB Chatbot
Contains entity models and database models
"""

from .entities import (
    Entity,
    EntityType,
    ExtractionMethod,
    ExtractionResult,
    ProgramEntity,
    DepartmentEntity,
    FeeEntity,
    CourseEntity,
    DateEntity,
    InfoTypeEntity,
    PROGRAM_ABBREVIATIONS,
    DEPARTMENT_ABBREVIATIONS,
    INFO_TYPE_MAPPINGS
)

__all__ = [
    'Entity',
    'EntityType',
    'ExtractionMethod',
    'ExtractionResult',
    'ProgramEntity',
    'DepartmentEntity',
    'FeeEntity',
    'CourseEntity',
    'DateEntity',
    'InfoTypeEntity',
    'PROGRAM_ABBREVIATIONS',
    'DEPARTMENT_ABBREVIATIONS',
    'INFO_TYPE_MAPPINGS'
]
