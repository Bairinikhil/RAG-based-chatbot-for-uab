"""
Program Service for UAB Chat Bot
Handles in-memory program index for fast program queries
"""

import json
import os
from typing import List, Dict, Optional
from collections import defaultdict

class ProgramService:
    """Service for handling program queries using in-memory index"""
    
    def __init__(self):
        """Initialize program service with in-memory program index"""
        self.programs = []
        self.programs_by_school = defaultdict(list)
        self.programs_by_level = defaultdict(list)
        self.programs_by_degree_type = defaultdict(list)
        self._load_programs()
    
    def _load_programs(self):
        """Load program index into memory at startup"""
        try:
            # Try to load from the main directory first
            program_file = 'program_index.json'
            if not os.path.exists(program_file):
                # Try the scraped data clean files directory
                program_file = 'scraped data clean files/program_index.json'
            
            if not os.path.exists(program_file):
                print(f"❌ Program index file not found: {program_file}")
                return
            
            with open(program_file, 'r', encoding='utf-8') as f:
                self.programs = json.load(f)
            
            # Build indexes for fast lookups
            self._build_indexes()
            
            print(f"✅ Loaded {len(self.programs)} programs into memory")
            print(f"📊 Schools: {len(self.programs_by_school)}")
            print(f"📊 Levels: {list(self.programs_by_level.keys())}")
            
        except Exception as e:
            print(f"❌ Error loading program index: {e}")
            self.programs = []
    
    def _build_indexes(self):
        """Build indexes for fast program lookups"""
        for program in self.programs:
            school = program.get('school', '')
            level = program.get('level', '')
            degree_type = program.get('degree_type', '')
            
            if school:
                self.programs_by_school[school].append(program)
            if level:
                self.programs_by_level[level].append(program)
            if degree_type:
                self.programs_by_degree_type[degree_type].append(program)
    
    def get_all_programs(self) -> List[Dict]:
        """Get all programs"""
        return self.programs
    
    def get_programs_by_school(self, school: str) -> List[Dict]:
        """Get all programs in a specific school"""
        return self.programs_by_school.get(school, [])
    
    def get_programs_by_level(self, level: str) -> List[Dict]:
        """Get all programs at a specific level (Undergraduate/Graduate)"""
        return self.programs_by_level.get(level, [])
    
    def get_programs_by_degree_type(self, degree_type: str) -> List[Dict]:
        """Get all programs of a specific degree type"""
        return self.programs_by_degree_type.get(degree_type, [])
    
    def search_programs(self, query: str) -> List[Dict]:
        """Search programs by name (case-insensitive)"""
        if not query:
            return []
        
        query_lower = query.lower()
        results = []
        
        for program in self.programs:
            program_name = program.get('program_name', '').lower()
            if query_lower in program_name:
                results.append(program)
        
        return results
    
    def get_programs_by_school_and_level(self, school: str, level: str) -> List[Dict]:
        """Get programs filtered by both school and level"""
        school_programs = self.get_programs_by_school(school)
        return [p for p in school_programs if p.get('level', '').lower() == level.lower()]
    
    def get_program_by_name(self, program_name: str) -> Optional[Dict]:
        """Get a specific program by exact name match"""
        for program in self.programs:
            if program.get('program_name', '').lower() == program_name.lower():
                return program
        return None
    
    def get_schools(self) -> List[str]:
        """Get list of all schools"""
        return list(self.programs_by_school.keys())
    
    def get_levels(self) -> List[str]:
        """Get list of all levels"""
        return list(self.programs_by_level.keys())
    
    def get_degree_types(self) -> List[str]:
        """Get list of all degree types"""
        return list(self.programs_by_degree_type.keys())
    
    def get_stats(self) -> Dict:
        """Get program statistics"""
        return {
            "total_programs": len(self.programs),
            "total_schools": len(self.programs_by_school),
            "total_levels": len(self.programs_by_level),
            "total_degree_types": len(self.programs_by_degree_type),
            "schools": list(self.programs_by_school.keys()),
            "levels": list(self.programs_by_level.keys()),
            "degree_types": list(self.programs_by_degree_type.keys())
        }
    
    def is_available(self) -> bool:
        """Check if program service is available"""
        return len(self.programs) > 0







