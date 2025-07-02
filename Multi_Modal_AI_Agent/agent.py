import os
import base64
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import requests
from PIL import Image
import io
import logging
from semanticscholar import SemanticScholar
import arxiv
from crossref.restful import Works

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningEngine:
    """Enhanced reasoning capabilities for the AI agent"""
    
    def __init__(self):
        self.reasoning_modes = {
            "standard": self._standard_reasoning,
            "chain_of_thought": self._chain_of_thought,
            "scientific": self._scientific_reasoning,
            "clinical": self._clinical_reasoning
        }
    
    def _standard_reasoning(self, prompt: str) -> str:
        """Default reasoning template"""
        return prompt
    
    def _chain_of_thought(self, prompt: str) -> str:
        """Step-by-step reasoning template"""
        return f"""
        Let's analyze this systematically:

        1. Problem Understanding: What is the core question in '{prompt}'?
        2. Information Extraction: What key information is available?
        3. Hypothesis Generation: What possible explanations exist?
        4. Evidence Evaluation: What supports or contradicts each hypothesis?
        5. Conclusion: What is the most supported conclusion?
        6. Limitations: What uncertainties remain?

        Now analyze: {prompt}
        """
    
    def _scientific_reasoning(self, prompt: str) -> str:
        """Scientific method-based reasoning"""
        return f"""
        Apply the scientific method:

        Observation: What phenomena does '{prompt}' describe?
        Question: What specific research questions emerge?
        Hypothesis: What testable predictions can we make?
        Experiment: How could we test these hypotheses?
        Analysis: What would constitute valid evidence?
        Conclusion: What would the results imply?

        Research analysis: {prompt}
        """
    
    def _clinical_reasoning(self, prompt: str) -> str:
        """Clinical decision-making framework"""
        return f"""
        Clinical analysis approach:

        1. Presentation: What are the key symptoms or findings?
        2. Differential Diagnosis: What conditions could explain this?
        3. Investigation: What tests would help differentiate?
        4. Evaluation: What do the results suggest?
        5. Management: What interventions are appropriate?
        6. Follow-up: What monitoring is needed?

        Clinical case: {prompt}
        """
    
    def apply_reasoning(self, prompt: str, mode: str = "standard") -> str:
        """Apply selected reasoning framework"""
        if mode not in self.reasoning_modes:
            logger.warning(f"Unknown reasoning mode {mode}, using standard")
            mode = "standard"
        return self.reasoning_modes[mode](prompt)

class CitationSearch:
    """Academic reference search and retrieval system"""
    
    def __init__(self):
        self.sch = SemanticScholar()
        self.crossref = Works()
        self.arxiv_client = arxiv.Client()
    
    def search_semantic_scholar(self, query: str, limit: int = 3) -> List[Dict]:
        """Search Semantic Scholar database"""
        try:
            results = self.sch.search_paper(query, limit=limit)
            return [{
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'year': paper.year,
                'abstract': paper.abstract,
                'url': paper.url,
                'source': 'Semantic Scholar'
            } for paper in results]
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []
    
    def search_arxiv(self, query: str, limit: int = 3) -> List[Dict]:
        """Search arXiv preprint database"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results = list(self.arxiv_client.results(search))
            return [{
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'year': result.published.year,
                'abstract': result.summary,
                'url': result.entry_id,
                'source': 'arXiv'
            } for result in results]
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []
    
    def search_crossref(self, query: str, limit: int = 3) -> List[Dict]:
        """Search Crossref for published works"""
        try:
            results = self.crossref.query(query).sample(limit)
            return [{
                'title': item.get('title', ['Untitled'])[0],
                'authors': [creator.get('given', '') + ' ' + creator.get('family', '') 
                           for creator in item.get('author', [])],
                'year': item.get('created', {}).get('date-parts', [[None]])[0][0],
                'abstract': item.get('abstract', 'No abstract available'),
                'url': item.get('URL', '#'),
                'source': 'Crossref'
            } for item in results if item]
        except Exception as e:
            logger.error(f"Crossref search failed: {e}")
            return []
    
    def find_references(self, query: str, limit_per_source: int = 2) -> List[Dict]:
        """Search across all academic sources"""
        references = []
        references.extend(self.search_semantic_scholar(query, limit_per_source))
        references.extend(self.search_arxiv(query, limit_per_source))
        references.extend(self.search_crossref(query, limit_per_source))
        return references
    
    def format_citations(self, references: List[Dict]) -> str:
        """Format references for inclusion in reports"""
        if not references:
            return "\n\n## References\nNo relevant references found."
        
        citation_text = "\n\n## References\n"
        for i, ref in enumerate(references, 1):
            authors = ref.get('authors', [])
            author_text = authors[0] + " et al." if len(authors) > 1 else authors[0] if authors else "Unknown"
            
            citation_text += (
                f"{i}. **{ref.get('title', 'Untitled')}**  \n"
                f"    *{author_text}* ({ref.get('year', 'n.d.')})  \n"
                f"    {ref.get('abstract', '')[:200]}...  \n"
                f"    [Read more]({ref.get('url', '#')}) | Source: {ref.get('source', 'Unknown')}\n\n"
            )
        return citation_text

class MultiModalAgent:
    """
    Enhanced multi-modal AI agent with reasoning and citation capabilities
    using Qwen2.5-VL-72B-Instruct via Nebius API
    """

    def __init__(self, api_key: str, base_url: str = "https://api.studio.nebius.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = "Qwen/Qwen2.5-VL-72B-Instruct"
        self.reasoning_engine = ReasoningEngine()
        self.citation_search = CitationSearch()

        # Report templates
        self.report_templates = {
            "comprehensive": self._comprehensive_template,
            "technical": self._technical_template,
            "executive": self._executive_template,
            "research": self._research_template,
            "reasoned": self._reasoned_template,
            "scientific": self._scientific_template,
            "clinical": self._clinical_template
        }

    def _encode_image(self, image_data: bytes) -> str:
        """Encode image to base64"""
        return base64.b64encode(image_data).decode('utf-8')

    def _prepare_messages(self, prompt: str, images: Optional[List[bytes]] = None) -> List[Dict]:
        """Prepare messages for the API call"""
        content = [{"type": "text", "text": prompt}]

        if images:
            for img_data in images:
                img_b64 = self._encode_image(img_data)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                })

        return [{"role": "user", "content": content}]

    def _call_nebius_api(self, messages: List[Dict], max_tokens: int = 4000) -> str:
        """Make API call to Nebius/Qwen model"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {e}")
            raise Exception(f"Failed to get response from AI model: {e}")

    # Report templates with enhanced reasoning
    def _comprehensive_template(self, base_prompt: str) -> str:
        return f"""
        Analyze the provided content comprehensively and generate a detailed report.

        Original Request: {base_prompt}

        # Executive Summary
        [Brief overview of key findings]

        # Detailed Analysis
        [Thorough examination of all aspects with logical reasoning]

        # Visual Analysis (if images provided)
        [Describe what you see in images, key elements, patterns, etc.]

        # Evidence-Based Insights
        [Supported by available data and references]

        # Recommendations
        [Actionable suggestions based on analysis]

        # Technical Details
        [Relevant technical information]

        # Conclusion
        [Summary and final thoughts]
        """

    def _technical_template(self, base_prompt: str) -> str:
        return f"""
        Technical Analysis Framework:

        Request: {base_prompt}

        # Technical Specifications
        # System Architecture
        # Performance Metrics
        # Implementation Details
        # Validation Methods
        # Technical References
        # Optimization Recommendations

        Focus on technical accuracy with supporting evidence.
        """

    def _executive_template(self, base_prompt: str) -> str:
        return f"""
        Executive Report Structure:

        Request: {base_prompt}

        # Business Impact (Quantified)
        # Strategic Recommendations
        # Cost-Benefit Analysis
        # Risk Assessment (Probability/Impact)
        # Implementation Roadmap
        # Key Performance Indicators

        Present concisely with data-driven insights.
        """

    def _research_template(self, base_prompt: str) -> str:
        return f"""
        Research Analysis Protocol:

        Topic: {base_prompt}

        # Literature Review
        # Methodology
        # Results
        # Discussion
        # Limitations
        # Future Research Directions
        # References

        Maintain academic rigor with proper citations.
        """

    def _reasoned_template(self, base_prompt: str) -> str:
        return self.reasoning_engine.apply_reasoning(base_prompt, "chain_of_thought")

    def _scientific_template(self, base_prompt: str) -> str:
        return self.reasoning_engine.apply_reasoning(base_prompt, "scientific")

    def _clinical_template(self, base_prompt: str) -> str:
        return self.reasoning_engine.apply_reasoning(base_prompt, "clinical")

    def process_request(
        self,
        prompt: str,
        images: Optional[List[bytes]] = None,
        report_type: str = "comprehensive",
        reasoning_mode: str = "standard",
        include_citations: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced processing with reasoning and citations
        """
        try:
            # Apply reasoning framework
            reasoned_prompt = self.reasoning_engine.apply_reasoning(prompt, reasoning_mode)

            # Apply report template
            if report_type in self.report_templates:
                enhanced_prompt = self.report_templates[report_type](reasoned_prompt)
            else:
                enhanced_prompt = reasoned_prompt

            # Prepare messages
            messages = self._prepare_messages(enhanced_prompt, images)

            # Get AI response
            logger.info(f"Processing {report_type} report with {reasoning_mode} reasoning")
            response = self._call_nebius_api(messages, max_tokens=6000)

            # Add citations if requested
            if include_citations and report_type in ["research", "technical", "scientific"]:
                references = self.citation_search.find_references(prompt)
                response += self.citation_search.format_citations(references)

            # Prepare result
            result = {
                "success": True,
                "report": response,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "report_type": report_type,
                    "reasoning_mode": reasoning_mode,
                    "model": self.model,
                    "has_images": bool(images),
                    "num_images": len(images) if images else 0,
                    "original_prompt": prompt,
                    "citation_count": len(references) if include_citations else 0
                }
            }

            logger.info("Request processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "report_type": report_type,
                    "original_prompt": prompt
                }
            }

    def analyze_image(self, image_data: bytes, prompt: str = "Describe this image in detail") -> Dict[str, Any]:
        """Enhanced image analysis with reasoning"""
        return self.process_request(
            prompt,
            images=[image_data],
            report_type="comprehensive",
            reasoning_mode="chain_of_thought"
        )

    def batch_analyze(self, requests: List[Dict]) -> List[Dict[str, Any]]:
        """Process multiple requests with consistent reasoning"""
        results = []
        for req in requests:
            result = self.process_request(
                req.get("prompt", ""),
                images=req.get("images"),
                report_type=req.get("report_type", "comprehensive"),
                reasoning_mode=req.get("reasoning_mode", "standard"),
                include_citations=req.get("include_citations", True)
            )
            results.append(result)
        return results

    def search_references(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Standalone reference search"""
        try:
            references = self.citation_search.find_references(query, limit)
            return {
                "success": True,
                "references": references,
                "count": len(references),
                "query": query
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }