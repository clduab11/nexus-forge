"""
Citation Agent for ParallaxMind

This agent is responsible for generating and managing citations for research sources,
supporting multiple citation styles, and creating bibliographies.
"""

import json
import logging
import uuid
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from google.cloud.aiplatform.adk import Agent, AgentContext, Task, action

# Import schemas
from schemas.agent_messages import (
    CitationRequest, CitationResponse, Source
)

# Import config
from adk_config import get_config

# Set up logging
logger = logging.getLogger(__name__)

class CitationAgent(Agent):
    """
    Citation Agent for ParallaxMind
    
    This agent is responsible for generating properly formatted citations for research
    sources across multiple citation styles (APA, MLA, Chicago, etc.) and creating
    comprehensive bibliographies.
    """
    
    def __init__(self):
        """Initialize citation agent with configuration."""
        super().__init__()
        self.config = get_config()
        self.logger = logging.getLogger("citation_agent")
        
        # Define supported citation styles
        self.citation_styles = ["apa", "mla", "chicago", "harvard", "ieee"]
        
        # Citation templates
        self.citation_templates = {
            "apa": {
                "article": "{author} ({year}). {title}. {journal}, {volume}({issue}), {pages}. {doi}",
                "website": "{author} ({year}). {title}. {site_name}. Retrieved {access_date}, from {url}",
                "book": "{author} ({year}). {title}. {publisher}.",
                "default": "{author} ({year}). {title}. Retrieved from {url}"
            },
            "mla": {
                "article": "{author}. \"{title}.\" {journal}, vol. {volume}, no. {issue}, {year}, pp. {pages}. {doi}",
                "website": "{author}. \"{title}.\" {site_name}, {publication_date}, {url}. Accessed {access_date}.",
                "book": "{author}. {title}. {publisher}, {year}.",
                "default": "{author}. \"{title}.\" {url}. Accessed {access_date}."
            },
            "chicago": {
                "article": "{author}. \"{title}.\" {journal} {volume}, no. {issue} ({year}): {pages}. {doi}",
                "website": "{author}. \"{title}.\" {site_name}. {publication_date}. {url}.",
                "book": "{author}. {title}. {publisher}, {year}.",
                "default": "{author}. \"{title}.\" Accessed {access_date}. {url}."
            }
        }
    
    def initialize(self, context: AgentContext) -> None:
        """Initialize the agent with context."""
        self.context = context
        self.logger.info("Citation agent initialized")
    
    @action
    def generate_citations(self, request: Dict) -> Dict:
        """
        Generate citations for a list of sources.
        
        Args:
            request: The citation request containing sources and style
            
        Returns:
            Dict containing sources with citations and a bibliography
        """
        try:
            # Parse request
            if isinstance(request, str):
                try:
                    request = json.loads(request)
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse request as JSON")
                    return self._create_error_response(str(uuid.uuid4()))
            
            request_id = request.get("request_id", str(uuid.uuid4()))
            sources = request.get("sources", [])
            style = request.get("style", "apa").lower()
            
            self.logger.info(f"Generating citations for {len(sources)} sources in {style} style")
            
            # Validate citation style
            if style not in self.citation_styles:
                self.logger.warning(f"Unsupported citation style: {style}. Defaulting to APA.")
                style = "apa"
            
            # Process each source
            processed_sources = []
            for source in sources:
                processed_source = self._generate_citation_for_source(source, style)
                processed_sources.append(processed_source)
            
            # Generate bibliography
            bibliography = self._generate_bibliography(processed_sources, style)
            
            # Create response
            response = {
                "request_id": request_id,
                "sources": [s.dict() if hasattr(s, 'dict') else s for s in processed_sources],
                "bibliography": bibliography,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating citations: {str(e)}")
            return self._create_error_response(request.get("request_id", str(uuid.uuid4())))
    
    def _generate_citation_for_source(self, source: Dict, style: str) -> Dict:
        """
        Generate a citation for a single source.
        
        Args:
            source: The source to cite
            style: The citation style to use
            
        Returns:
            Source with added citation
        """
        try:
            # Convert to Source object if it's a dict
            if isinstance(source, dict):
                source_obj = Source(**source)
            else:
                source_obj = source
            
            # Generate citation based on available metadata
            source_type = self._determine_source_type(source_obj)
            
            # Try using templates first (for simple cases)
            citation = self._generate_citation_from_template(source_obj, style, source_type)
            
            # If template generation fails or for complex sources, use the AI model
            if not citation or len(citation) < 20:
                citation = self._generate_citation_with_ai(source_obj, style)
            
            # Update the source with the citation
            if isinstance(source, dict):
                source["citation"] = citation
                return source
            else:
                source_obj.citation = citation
                return source_obj
            
        except Exception as e:
            self.logger.error(f"Error generating citation for source: {str(e)}")
            
            # Return original source with error note
            if isinstance(source, dict):
                source["citation"] = f"[Citation Error: {str(e)}]"
                return source
            else:
                source.citation = f"[Citation Error: {str(e)}]"
                return source
    
    def _determine_source_type(self, source: Source) -> str:
        """
        Determine the type of source based on its attributes.
        
        Args:
            source: The source to analyze
            
        Returns:
            Source type: article, website, book, or default
        """
        url = source.url if hasattr(source, 'url') else ""
        title = source.title if hasattr(source, 'title') else ""
        
        # Check for journal articles
        if hasattr(source, 'journal') and source.journal:
            return "article"
        
        # Check URL patterns for websites
        if url:
            # If URL contains typical website domains
            if any(domain in url.lower() for domain in [".com", ".org", ".net", ".gov", ".edu"]):
                return "website"
        
        # Check for book indicators
        if hasattr(source, 'publisher') and source.publisher:
            return "book"
        
        # Default to website for most online sources
        return "website"
    
    def _generate_citation_from_template(self, source: Source, style: str, source_type: str) -> str:
        """
        Generate a citation using pre-defined templates.
        
        Args:
            source: The source to cite
            style: The citation style to use
            source_type: The type of source
            
        Returns:
            Formatted citation or empty string if template fails
        """
        try:
            # Get the appropriate template
            templates = self.citation_templates.get(style, self.citation_templates["apa"])
            template = templates.get(source_type, templates["default"])
            
            # Extract source attributes
            attrs = {
                "author": getattr(source, "author", ""),
                "title": getattr(source, "title", ""),
                "year": "",
                "publication_date": "",
                "journal": "",
                "volume": "",
                "issue": "",
                "pages": "",
                "publisher": "",
                "site_name": getattr(source, "site_name", ""),
                "url": getattr(source, "url", ""),
                "doi": "",
                "access_date": getattr(source, "access_date", datetime.now().strftime("%Y-%m-%d"))
            }
            
            # Process publication date to extract year
            if hasattr(source, "publication_date") and source.publication_date:
                attrs["publication_date"] = source.publication_date
                # Try to extract year from publication date
                year_match = re.search(r'(\d{4})', source.publication_date)
                if year_match:
                    attrs["year"] = year_match.group(1)
                else:
                    attrs["year"] = "n.d."  # no date
            else:
                attrs["year"] = "n.d."
            
            # Format author name properly
            if attrs["author"]:
                # Check if multiple authors (separated by commas or 'and')
                if "," in attrs["author"] or " and " in attrs["author"].lower():
                    # Multiple authors - keep as is for now (complex formatting handled by AI model)
                    pass
                else:
                    # Single author - format based on style
                    name_parts = attrs["author"].split()
                    if len(name_parts) > 1 and style == "apa":
                        # APA: Last name, F. M.
                        last_name = name_parts[-1]
                        initials = "".join([f"{n[0]}." for n in name_parts[:-1]])
                        attrs["author"] = f"{last_name}, {initials}"
                    elif len(name_parts) > 1 and style == "mla":
                        # MLA: Last name, First name
                        last_name = name_parts[-1]
                        first_name = " ".join(name_parts[:-1])
                        attrs["author"] = f"{last_name}, {first_name}"
            else:
                # No author provided
                attrs["author"] = "Unknown"
            
            # Format the citation using the template
            citation = template
            for key, value in attrs.items():
                placeholder = "{" + key + "}"
                if placeholder in citation:
                    citation = citation.replace(placeholder, value if value else "")
            
            # Clean up any remaining placeholders or empty parentheses
            citation = re.sub(r'\(\s*\)', '', citation)
            citation = re.sub(r'\[\s*\]', '', citation)
            citation = re.sub(r'\s+,', ',', citation)
            citation = re.sub(r'\s+\.', '.', citation)
            citation = re.sub(r'\s{2,}', ' ', citation)
            
            return citation.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating citation from template: {str(e)}")
            return ""
    
    def _generate_citation_with_ai(self, source: Source, style: str) -> str:
        """
        Generate a citation using AI for complex cases.
        
        Args:
            source: The source to cite
            style: The citation style to use
            
        Returns:
            AI-generated citation
        """
        try:
            # Create prompt for citation generation
            citation_prompt = f"""
            Generate a correctly formatted citation in {style.upper()} style for the following source:
            
            Title: {getattr(source, 'title', 'Unknown Title')}
            URL: {getattr(source, 'url', '')}
            Author: {getattr(source, 'author', 'Unknown Author')}
            Publication Date: {getattr(source, 'publication_date', 'n.d.')}
            Website/Publisher: {getattr(source, 'site_name', '')}
            Access Date: {getattr(source, 'access_date', datetime.now().strftime("%Y-%m-%d"))}
            
            Return ONLY the formatted citation, nothing else.
            """
            
            # Use vertex_ai_generate tool to generate citation
            citation_task = Task(
                tool_name="vertex_ai_generate",
                task_input={
                    "model": self.config.primary_model,
                    "prompt": citation_prompt
                }
            )
            
            # Execute citation task
            citation_result = self.context.execute_task(citation_task)
            
            if not citation_result or not isinstance(citation_result, str):
                return self._create_fallback_citation(source, style)
            
            return citation_result.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating citation with AI: {str(e)}")
            return self._create_fallback_citation(source, style)
    
    def _create_fallback_citation(self, source: Source, style: str) -> str:
        """
        Create a fallback citation when other methods fail.
        
        Args:
            source: The source to cite
            style: The citation style to use
            
        Returns:
            Basic fallback citation
        """
        # Get basic source attributes
        title = getattr(source, "title", "Unknown Title")
        url = getattr(source, "url", "")
        site_name = getattr(source, "site_name", "")
        access_date = getattr(source, "access_date", datetime.now().strftime("%Y-%m-%d"))
        
        # Create a basic citation based on style
        if style == "apa":
            return f"{title}. Retrieved {access_date}, from {url}"
        elif style == "mla":
            return f"\"{title}.\" {site_name if site_name else url}. Accessed {access_date}."
        elif style == "chicago":
            return f"\"{title}.\" Accessed {access_date}. {url}."
        elif style == "harvard":
            return f"{title}. [Online] Available at: {url} [Accessed {access_date}]."
        elif style == "ieee":
            return f"[1] \"{title},\" {site_name if site_name else url}. [Online]. Available: {url}."
        else:
            return f"{title}. {url}. Accessed {access_date}."
    
    def _generate_bibliography(self, sources: List, style: str) -> str:
        """
        Generate a complete bibliography from all sources.
        
        Args:
            sources: List of sources with citations
            style: The citation style to use
            
        Returns:
            Formatted bibliography
        """
        try:
            # Extract citations from sources
            citations = []
            for source in sources:
                if isinstance(source, dict):
                    citation = source.get("citation", "")
                else:
                    citation = getattr(source, "citation", "")
                
                if citation:
                    citations.append(citation)
            
            if not citations:
                return "No sources available for bibliography."
            
            # Create bibliography based on style
            bibliography = f"Bibliography ({style.upper()} Style)\n\n"
            
            # Sort citations alphabetically (typical for most styles)
            sorted_citations = sorted(citations, key=lambda x: re.sub(r'^\W+', '', x).lower())
            
            # Format based on style
            if style == "apa" or style == "harvard":
                # For APA and Harvard, hanging indent is typical
                for citation in sorted_citations:
                    bibliography += f"{citation}\n\n"
            elif style == "mla" or style == "chicago":
                # For MLA and Chicago, hanging indent is typical
                for citation in sorted_citations:
                    bibliography += f"{citation}\n\n"
            elif style == "ieee":
                # For IEEE, numbered list
                for i, citation in enumerate(sorted_citations, 1):
                    # Remove any existing numbering
                    clean_citation = re.sub(r'^\[\d+\]\s*', '', citation)
                    bibliography += f"[{i}] {clean_citation}\n\n"
            else:
                # Default format
                for citation in sorted_citations:
                    bibliography += f"{citation}\n\n"
            
            return bibliography.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating bibliography: {str(e)}")
            
            # Create a simple fallback bibliography
            bibliography = f"Bibliography ({style.upper()} Style)\n\n"
            for i, source in enumerate(sources, 1):
                title = ""
                url = ""
                
                if isinstance(source, dict):
                    title = source.get("title", "")
                    url = source.get("url", "")
                else:
                    title = getattr(source, "title", "")
                    url = getattr(source, "url", "")
                
                bibliography += f"{i}. {title}. {url}\n\n"
            
            return bibliography.strip()
    
    def _create_error_response(self, request_id: str) -> Dict:
        """
        Create an error response when citation generation fails.
        
        Args:
            request_id: The request ID
            
        Returns:
            Dict containing error information
        """
        return {
            "request_id": request_id,
            "sources": [],
            "bibliography": "Unable to generate citations due to an error.",
            "timestamp": datetime.now().isoformat()
        }