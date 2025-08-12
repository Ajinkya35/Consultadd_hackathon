import google.generativeai as genai
from typing import Dict
import json
import os
from dotenv import load_dotenv

class LLMEngine:
    def __init__(self):
        """Initialize Gemini API"""
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.debug = True  # Add debug flag
        
    async def analyze_rfp(self, rfp_text: str) -> Dict:
        """Analyze RFP using Gemini"""
        try:
            analysis_prompt = f"""
            Analyze this RFP and list requirements in these exact sections:

            === ELIGIBILITY REQUIREMENTS ===
            - [list each requirement on a new line starting with -]

            === COMPLIANCE REQUIREMENTS ===
            - [list each requirement on a new line starting with -]

            === SUBMISSION REQUIREMENTS ===
            - [list each requirement on a new line starting with -]

            === RISK FACTORS ===
            - [list each risk on a new line starting with -]

            RFP Text:
            {rfp_text}
            """

            response = await self.model.generate_content_async(
                analysis_prompt,
                generation_config={
                    'temperature': 0.3,
                    'max_output_tokens': 2048,
                }
            )
            
            if not response or not response.text:
                return self._get_error_response("Empty response from model")

            if self.debug:
                print("\nDEBUG - Raw Response:")
                print("-" * 50)
                print(response.text)
                print("-" * 50)

            parsed_response = self._parse_text_response(response.text)
            if not parsed_response.get("eligibility_criteria"):
                return self._get_error_response("No eligibility criteria found")
                
            return parsed_response
            
        except Exception as e:
            print(f"Analysis Error: {str(e)}")
            return self._get_error_response(str(e))
    
    def _parse_text_response(self, text: str) -> Dict:
        """Parse text response into structured format"""
        try:
            sections = {
                "eligibility_criteria": [],
                "compliance_requirements": [],
                "submission_requirements": [],
                "risk_analysis": []
            }
            
            current_section = None
            lines = text.split('\n')
            
            # Map different possible section headers to internal keys
            section_mapping = {
                "ELIGIBILITY REQUIREMENTS": "eligibility_criteria",
                "COMPLIANCE REQUIREMENTS": "compliance_requirements",
                "SUBMISSION REQUIREMENTS": "submission_requirements",
                "RISK FACTORS": "risk_analysis"
            }
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers more flexibly
                for header, section_key in section_mapping.items():
                    if header in line.upper():
                        current_section = section_key
                        break
                
                # Add content if we're in a section and line starts with - or •
                if current_section and (line.startswith('-') or line.startswith('•')):
                    clean_line = line.lstrip('-• ').strip()
                    if clean_line and not any(ignore in clean_line.lower() for ignore in 
                        ["list each requirement", "list each risk"]):
                        sections[current_section].append(clean_line)
            
            if self.debug:
                print("\nDEBUG - Parsed Sections:")
                for section, items in sections.items():
                    print(f"\n{section}:")
                    for item in items:
                        print(f"  - {item}")

            # Convert sections to final format with validation
            result = {}
            for key, items in sections.items():
                if items:
                    result[key] = "\n• " + "\n• ".join(items)
                else:
                    result[key] = "No items identified"
                    
            return result
            
        except Exception as e:
            print(f"Parsing Error: {str(e)}")
            print(f"Current section: {current_section}")
            print(f"Available sections: {sections.keys()}")
            return {
                "eligibility_criteria": f"Error occurred during parsing: {str(e)}",
                "compliance_requirements": "Error occurred during parsing",
                "submission_requirements": "Error occurred during parsing",
                "risk_analysis": "Error occurred during parsing"
            }

    async def check_eligibility(self, rfp_text: str, company_data: Dict) -> bool:
        """Check if company meets eligibility criteria"""
        try:
            eligibility_prompt = f"""
            TASK: Determine if the company is eligible to bid on this RFP.
            
            RFP Text:
            {rfp_text}

            Company Profile:
            {json.dumps(company_data, indent=2)}

            INSTRUCTIONS:
            1. Start with YES or NO
            2. List matching requirements
            3. List any gaps or concerns
            """

            response = await self.model.generate_content_async(
                eligibility_prompt,
                generation_config={
                    'temperature': 0.3,
                    'max_output_tokens': 1024
                }
            )
            
            if not response or not response.text:
                print("Error: Empty response from model")
                return False

            result = response.text.upper().strip()
            print("\nEligibility Analysis:")
            print("-" * 50)
            print(response.text)
            print("-" * 50)
            
            return result.startswith("YES")
            
        except Exception as e:
            print(f"Eligibility Check Error: {str(e)}")
            return False

    def _get_error_response(self, error_msg: str) -> Dict:
        """Generate error response structure"""
        return {
            "eligibility_criteria": f"Analysis error: {error_msg}",
            "compliance_requirements": "Analysis failed",
            "submission_requirements": "Analysis failed",
            "risk_analysis": "Analysis failed"
        }