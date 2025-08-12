from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
from typing import List, Dict
import json

class GeminiEngine:
    def __init__(self, project_id: str, location: str):
        """Initialize Gemini client"""
        self.client = aiplatform.gapic.PublisherClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        )
        self.project_id = project_id
        self.location = location
        
    def _create_prompt(self, text: str, query: str) -> str:
        """Create prompt for Gemini"""
        return f"""Analyze the following RFP text and answer the query:

RFP Text:
{text}

Query:
{query}

Please provide a detailed analysis focused specifically on the query.
"""

    def analyze_rfp(self, rfp_text: str) -> Dict:
        """Analyze RFP using Gemini"""
        # Define key analysis areas
        analysis_queries = {
            "eligibility": "What are the mandatory eligibility criteria and requirements to bid?",
            "compliance": "What are the key compliance requirements and potential deal-breakers?",
            "submission": "What are the detailed submission requirements including format, deadlines etc?",
            "risks": "What are the potential contract risks and unfavorable clauses?"
        }
        
        results = {}
        
        for category, query in analysis_queries.items():
            prompt = self._create_prompt(rfp_text, query)
            
            # Call Gemini API
            response = self.client.predict_text(
                project=self.project_id,
                location=self.location,
                prompt=prompt,
                temperature=0.3,
                max_output_tokens=1024
            )
            
            results[category] = response.text
            
        return results

    def check_eligibility(self, rfp_text: str, company_data: Dict) -> bool:
        """Check if company is eligible to bid"""
        prompt = f"""
        Based on the RFP requirements and company data below, determine if the company is eligible to bid.
        
        RFP Text:
        {rfp_text}
        
        Company Data:
        {json.dumps(company_data, indent=2)}
        
        Please respond with a clear YES or NO and explain the reasoning.
        """
        
        response = self.client.predict_text(
            project=self.project_id,
            location=self.location, 
            prompt=prompt,
            temperature=0.1
        )
        
        return "YES" in response.text.upper()