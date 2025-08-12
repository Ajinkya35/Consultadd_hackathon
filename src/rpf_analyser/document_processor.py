import PyMuPDF  # Change from import fitz
import re
from typing import Dict, List
from .llm_engine import LLMEngine

class DocumentProcessor:
    def __init__(self):
        self.document_text = ""
        self.sections = {}
        self.llm = LLMEngine()
    
    def load_pdf(self, file_path: str) -> None:
        """Load and process PDF document"""
        doc = PyMuPDF.open(file_path)  # Change from fitz.open
        text = ""
        for page in doc:
            text += page.get_text()
        self.document_text = text
        self._parse_sections()
    
    def _parse_sections(self) -> None:
        """Parse document into logical sections"""
        section_patterns = {
            "eligibility": r"(?i)(eligibility\s+requirements|qualification\s+criteria)",
            "submission": r"(?i)(submission\s+requirements|proposal\s+format)",
            "terms": r"(?i)(terms\s+and\s+conditions|contract\s+terms)"
        }
        
        for section_name, pattern in section_patterns.items():
            matches = re.finditer(pattern, self.document_text)
            for match in matches:
                start = match.start()
                # Find the next section or end of document
                next_section = float('inf')
                for other_pattern in section_patterns.values():
                    next_match = re.search(other_pattern, self.document_text[start+100:])
                    if next_match:
                        next_section = min(next_section, start + 100 + next_match.start())
                
                section_text = self.document_text[start:next_section if next_section != float('inf') else None]
                self.sections[section_name] = section_text.strip()

    def get_section(self, section_name: str) -> str:
        """Retrieve specific section content"""
        return self.sections.get(section_name, "")

    async def analyze_rfp(self) -> Dict:
        """Run comprehensive RFP analysis"""
        if not self.document_text:
            raise ValueError("No document loaded")
            
        analysis_results = await self.llm.analyze_rfp(self.document_text)
        
        return {
            "eligibility_criteria": analysis_results["eligibility"],
            "compliance_requirements": analysis_results["compliance"],
            "submission_requirements": analysis_results["submission"],
            "risk_analysis": analysis_results["risks"],
            "sections": self.sections
        }

    async def check_eligibility(self, company_data: Dict) -> bool:
        """Check if company meets eligibility criteria"""
        if not self.document_text:
            raise ValueError("No document loaded")
            
        return await self.llm.check_eligibility(
            self.document_text,
            company_data
        )