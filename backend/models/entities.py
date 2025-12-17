"""
Unified Entity Models for UAB Chatbot
Defines all entity types that can be extracted from user queries.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class EntityType(Enum):
    """All supported entity types in the system"""
    PROGRAM = "program"
    DEPARTMENT = "department"
    SCHOOL = "school"
    DEGREE_LEVEL = "degree_level"
    DEGREE_TYPE = "degree_type"
    INFO_TYPE = "info_type"
    FEE = "fee"
    COURSE = "course"
    DATE = "date"
    STUDENT_NAME = "student_name"
    STUDENT_ID = "student_id"
    FACULTY_NAME = "faculty_name"
    LOCATION = "location"
    SEMESTER = "semester"
    GENERAL = "general"


class ExtractionMethod(Enum):
    """Method used to extract the entity"""
    HEURISTIC = "heuristic"
    REGEX = "regex"
    LLM = "llm"
    NLP = "nlp"
    HYBRID = "hybrid"


@dataclass
class Entity:
    """
    Represents a single extracted entity from user query
    """
    value: str
    type: Optional[EntityType] = None
    normalized_value: Optional[str] = None
    confidence: float = 1.0  # 0.0 to 1.0
    method: ExtractionMethod = ExtractionMethod.HEURISTIC
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate entity after initialization"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        # Set normalized value if not provided
        if self.normalized_value is None:
            self.normalized_value = self.value.lower().strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return {
            "type": self.type.value,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "method": self.method.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "metadata": self.metadata
        }


@dataclass
class ProgramEntity(Entity):
    """Program-specific entity (e.g., MSECE, PhD in ECE)"""
    program_abbreviation: Optional[str] = None
    program_full_name: Optional[str] = None

    def __post_init__(self):
        self.type = EntityType.PROGRAM
        super().__post_init__()
        self.metadata.update({
            "abbreviation": self.program_abbreviation,
            "full_name": self.program_full_name
        })


@dataclass
class DepartmentEntity(Entity):
    """Department-specific entity"""
    department_abbreviation: Optional[str] = None

    def __post_init__(self):
        self.type = EntityType.DEPARTMENT
        super().__post_init__()
        if self.department_abbreviation:
            self.metadata["abbreviation"] = self.department_abbreviation


@dataclass
class FeeEntity(Entity):
    """Fee/cost entity (e.g., $1,200, tuition fee)"""
    amount: Optional[float] = None
    currency: str = "USD"
    fee_type: Optional[str] = None  # tuition, application, etc.

    def __post_init__(self):
        self.type = EntityType.FEE
        super().__post_init__()
        self.metadata.update({
            "amount": self.amount,
            "currency": self.currency,
            "fee_type": self.fee_type
        })


@dataclass
class CourseEntity(Entity):
    """Course entity (e.g., EE 660, MA 125)"""
    course_code: Optional[str] = None
    course_name: Optional[str] = None
    course_department: Optional[str] = None

    def __post_init__(self):
        self.type = EntityType.COURSE
        super().__post_init__()
        self.metadata.update({
            "code": self.course_code,
            "name": self.course_name,
            "department": self.course_department
        })


@dataclass
class DateEntity(Entity):
    """Date/deadline entity"""
    parsed_date: Optional[datetime] = None
    date_type: Optional[str] = None  # deadline, start_date, end_date
    is_relative: bool = False  # True for "next week", "tomorrow"

    def __post_init__(self):
        self.type = EntityType.DATE
        super().__post_init__()
        self.metadata.update({
            "parsed_date": self.parsed_date.isoformat() if self.parsed_date else None,
            "date_type": self.date_type,
            "is_relative": self.is_relative
        })


@dataclass
class InfoTypeEntity(Entity):
    """Information type entity (admission, requirements, etc.)"""
    category: Optional[str] = None

    def __post_init__(self):
        self.type = EntityType.INFO_TYPE
        super().__post_init__()
        if self.category:
            self.metadata["category"] = self.category


@dataclass
class ExtractionResult:
    """
    Complete result of entity extraction from a query
    """
    query: str
    entities: List[Entity] = field(default_factory=list)
    extraction_time_ms: float = 0.0
    used_llm: bool = False
    error: Optional[str] = None

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type"""
        return [e for e in self.entities if e.type == entity_type]

    def get_best_entity(self, entity_type: EntityType) -> Optional[Entity]:
        """Get the highest confidence entity of a specific type"""
        entities = self.get_entities_by_type(entity_type)
        return max(entities, key=lambda e: e.confidence) if entities else None

    def has_entity_type(self, entity_type: EntityType) -> bool:
        """Check if any entity of given type was extracted"""
        return any(e.type == entity_type for e in self.entities)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "entities": [e.to_dict() for e in self.entities],
            "extraction_time_ms": self.extraction_time_ms,
            "used_llm": self.used_llm,
            "error": self.error,
            "entity_count": len(self.entities)
        }

    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert to legacy format for backward compatibility
        Returns dict with: department_name, program_abbreviation, info_type
        """
        result = {
            "department_name": None,
            "program_abbreviation": None,
            "info_type": None,
            "school": None,
            "level": None,
            "degree_type": None
        }

        # Map to legacy format
        dept = self.get_best_entity(EntityType.DEPARTMENT)
        if dept:
            result["department_name"] = dept.value  # Use exact value, not normalized (lowercase)

        prog = self.get_best_entity(EntityType.PROGRAM)
        if prog:
            result["program_abbreviation"] = prog.metadata.get("abbreviation") or prog.normalized_value

        info = self.get_best_entity(EntityType.INFO_TYPE)
        if info:
            result["info_type"] = info.normalized_value

        school = self.get_best_entity(EntityType.SCHOOL)
        if school:
            result["school"] = school.normalized_value

        level = self.get_best_entity(EntityType.DEGREE_LEVEL)
        if level:
            result["level"] = level.normalized_value

        degree = self.get_best_entity(EntityType.DEGREE_TYPE)
        if degree:
            result["degree_type"] = degree.normalized_value

        return result


# Entity normalization mappings
PROGRAM_ABBREVIATIONS = {
    # Electrical and Computer Engineering
    "msece": "MSECE",
    "ms ece": "MSECE",
    "master of science in electrical and computer engineering": "MSECE",
    "masters in ece": "MSECE",

    # Civil Engineering
    "civil engineering ms": "civil_engineering_msce",
    "civil engineering master": "civil_engineering_msce",
    "msce civil": "civil_engineering_msce",
    "ms civil engineering": "civil_engineering_msce",
    "master of science in civil engineering": "civil_engineering_msce",
    "civil engineering phd": "civil_engineering_phd",
    "phd civil engineering": "civil_engineering_phd",

    # Computer Science
    "computer science ms": "computer_science_ms",
    "cs ms": "computer_science_ms",
    "ms computer science": "computer_science_ms",
    "master of science in computer science": "computer_science_ms",
    "computer science phd": "computer_science_phd",
    "cs phd": "computer_science_phd",
    "phd computer science": "computer_science_phd",

    # MBA
    "mba": "mba",
    "master of business administration": "mba",
    "business administration master": "mba",

    # Generic degree levels (fallback)
    "phd": "PhD",
    "ph.d": "PhD",
    "doctorate": "PhD",
    "doctoral": "PhD",
    "ms": "MS",
    "master": "MS",
    "masters": "MS"
}

DEPARTMENT_ABBREVIATIONS = {
    "electrical and computer engineering": "ECE",
    "ece": "ECE",
    "electrical engineering": "ECE",
    "computer engineering": "ECE",
    "civil engineering": "Civil Engineering",
    "computer science": "Computer Science",
    "cs": "Computer Science",
    "business": "Business",
    "business administration": "Business"
}

INFO_TYPE_MAPPINGS = {
    # Admission related - maps to "overview_admission" (database value)
    "admission": "overview_admission",
    "admissions": "overview_admission",
    "apply": "overview_admission",
    "application": "overview_admission",
    "requirements": "overview_admission",
    "eligibility": "overview_admission",
    "prerequisites": "overview_admission",
    "admission_requirements": "overview_admission",  # Legacy compatibility

    # Degree requirements - maps to "degree_requirements" (generic)
    "degree": "degree_requirements",
    "curriculum": "degree_requirements",
    "coursework": "degree_requirements",
    "courses": "degree_requirements",
    "credits": "degree_requirements",

    # Deadlines
    "deadline": "deadlines",
    "deadlines": "deadlines",
    "due date": "deadlines",
    "timeline": "deadlines",

    # Tuition and Fees - maps to "tuition_and_fees" (database value)
    "tuition": "tuition_and_fees",
    "tuition_and_fees": "tuition_and_fees",
    "cost": "tuition_and_fees",
    "fee": "tuition_and_fees",
    "fees": "tuition_and_fees",
    "price": "tuition_and_fees",
    "how much": "tuition_and_fees",
    "expensive": "tuition_and_fees",

    # Financial Aid - maps to "financial_aid" (database value)
    "financial aid": "financial_aid",
    "financial_aid": "financial_aid",
    "scholarship": "financial_aid",
    "scholarships": "financial_aid",
    "assistantship": "financial_aid",
    "assistantships": "financial_aid",
    "funding": "financial_aid",
    "stipend": "financial_aid",

    # Contact - maps to "contact_resources" (database value)
    "contact": "contact_resources",
    "email": "contact_resources",
    "phone": "contact_resources",
    "office": "contact_resources",
    "contact_info": "contact_resources",  # Legacy compatibility

    # Overview - maps to "overview_admission" (most common case)
    "overview": "overview_admission",
    "about": "overview_admission",
    "information": "overview_admission",
    "details": "overview_admission",
    "description": "overview_admission"
}
