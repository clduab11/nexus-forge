"""
Citation Agent for ParallaxMind ADK Implementation

This agent handles source tracking, verification, and citation management
for research results, ensuring credibility and proper attribution.
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class CitationAgent:
    """
    Specialized agent for managing citations and source verification.

    This agent is responsible for:
    - Tracking sources and their reliability
    - Generating proper citations
    - Verifying source credibility
    - Managing citation networks
    - Cross-referencing information
    """

    def __init__(self):
        """Initialize the Citation Agent."""
        self.citations_cache = {}
        self.source_reliability_scores = {}
        self.citation_networks = {}
        self.duplicate_detection_cache = {}

        # Citation format templates
        self.citation_formats = {
            "apa": "{authors} ({year}). {title}. {publication}. Retrieved from {url}",
            "mla": '{authors}. "{title}." {publication}, {date}, {url}.',
            "chicago": '{authors}. "{title}." {publication}. Accessed {access_date}. {url}.',
            "simple": "{title} - {domain} ({date})",
        }

        # Domain reliability patterns
        self.high_reliability_domains = {
            "edu",
            "gov",
            "org",
            "nature.com",
            "science.org",
            "pubmed.ncbi.nlm.nih.gov",
            "scholar.google.com",
            "arxiv.org",
            "ieee.org",
            "acm.org",
            "springer.com",
            "wiley.com",
            "elsevier.com",
            "tandfonline.com",
            "jstor.org",
        }

        self.medium_reliability_domains = {
            "reuters.com",
            "bbc.com",
            "npr.org",
            "pbs.org",
            "apnews.com",
            "nytimes.com",
            "washingtonpost.com",
            "theguardian.com",
            "economist.com",
        }

        logger.info("Citation Agent initialized")

    async def process_sources(self, sources: List[Dict], context: str = "") -> Dict:
        """
        Process and analyze a list of sources for citation management.

        Args:
            sources: List of source dictionaries with URL, title, content, etc.
            context: Research context for better citation analysis

        Returns:
            Dictionary containing processed citations and analysis
        """
        try:
            logger.info(f"Processing {len(sources)} sources for citation analysis")

            processed_citations = []
            source_network = {}
            reliability_analysis = {}
            duplicates_found = []

            for i, source in enumerate(sources):
                url = source.get("url", "")
                title = source.get("title", "Untitled")
                content = source.get("content", "")

                if not url:
                    continue

                # Generate citation ID
                citation_id = self._generate_citation_id(url, title)

                # Check for duplicates
                duplicate_info = await self._detect_duplicates(
                    citation_id, url, title, content
                )
                if duplicate_info:
                    duplicates_found.append(duplicate_info)
                    continue

                # Analyze source reliability
                reliability = await self._analyze_source_reliability(
                    url, title, content
                )
                reliability_analysis[citation_id] = reliability

                # Extract metadata
                metadata = await self._extract_metadata(source)

                # Generate citation formats
                citation_formats = await self._generate_citation_formats(metadata)

                # Build source network connections
                connections = await self._analyze_source_connections(
                    citation_id, url, content, processed_citations
                )
                source_network[citation_id] = connections

                # Create processed citation entry
                citation_entry = {
                    "id": citation_id,
                    "url": url,
                    "title": title,
                    "metadata": metadata,
                    "reliability_score": reliability["score"],
                    "reliability_factors": reliability["factors"],
                    "citation_formats": citation_formats,
                    "network_connections": connections,
                    "processing_timestamp": datetime.now().isoformat(),
                    "content_hash": (
                        hashlib.md5(content.encode()).hexdigest()[:16]
                        if content
                        else None
                    ),
                }

                processed_citations.append(citation_entry)

                # Cache the citation
                self.citations_cache[citation_id] = citation_entry

            # Analyze citation patterns
            citation_patterns = await self._analyze_citation_patterns(
                processed_citations, context
            )

            # Generate citation summary
            citation_summary = await self._generate_citation_summary(
                processed_citations, reliability_analysis
            )

            result = {
                "citations": processed_citations,
                "source_network": source_network,
                "reliability_analysis": reliability_analysis,
                "duplicates_found": duplicates_found,
                "citation_patterns": citation_patterns,
                "citation_summary": citation_summary,
                "total_sources": len(sources),
                "processed_sources": len(processed_citations),
                "high_reliability_count": len(
                    [c for c in processed_citations if c["reliability_score"] >= 8.0]
                ),
                "medium_reliability_count": len(
                    [
                        c
                        for c in processed_citations
                        if 5.0 <= c["reliability_score"] < 8.0
                    ]
                ),
                "low_reliability_count": len(
                    [c for c in processed_citations if c["reliability_score"] < 5.0]
                ),
            }

            logger.info(
                f"Processed {len(processed_citations)} citations with {len(duplicates_found)} duplicates found"
            )
            return result

        except Exception as e:
            logger.error(f"Error processing sources for citations: {e}")
            return {
                "citations": [],
                "error": str(e),
                "source_network": {},
                "reliability_analysis": {},
            }

    async def verify_citation_credibility(
        self, citation_id: str, cross_check_sources: List[Dict] = None
    ) -> Dict:
        """
        Verify the credibility of a specific citation through cross-referencing.

        Args:
            citation_id: ID of the citation to verify
            cross_check_sources: Additional sources to cross-check against

        Returns:
            Dictionary containing credibility verification results
        """
        try:
            if citation_id not in self.citations_cache:
                return {"error": "Citation not found", "verified": False}

            citation = self.citations_cache[citation_id]
            verification_result = {
                "citation_id": citation_id,
                "verified": False,
                "verification_score": 0.0,
                "verification_factors": [],
                "cross_references": [],
                "conflicts": [],
                "recommendations": [],
            }

            # Check domain credibility
            domain_score = await self._check_domain_credibility(citation["url"])
            verification_result["verification_factors"].append(
                {
                    "factor": "domain_credibility",
                    "score": domain_score,
                    "description": "Credibility based on domain reputation",
                }
            )

            # Check content authenticity indicators
            content_authenticity = await self._check_content_authenticity(citation)
            verification_result["verification_factors"].append(
                {
                    "factor": "content_authenticity",
                    "score": content_authenticity["score"],
                    "description": "Authenticity based on content analysis",
                    "details": content_authenticity["details"],
                }
            )

            # Cross-reference with other sources
            if cross_check_sources:
                cross_ref_result = await self._cross_reference_sources(
                    citation, cross_check_sources
                )
                verification_result["cross_references"] = cross_ref_result["matches"]
                verification_result["conflicts"] = cross_ref_result["conflicts"]
                verification_result["verification_factors"].append(
                    {
                        "factor": "cross_reference_consistency",
                        "score": cross_ref_result["consistency_score"],
                        "description": "Consistency with other sources",
                    }
                )

            # Calculate overall verification score
            if verification_result["verification_factors"]:
                total_score = sum(
                    factor["score"]
                    for factor in verification_result["verification_factors"]
                )
                verification_result["verification_score"] = total_score / len(
                    verification_result["verification_factors"]
                )
                verification_result["verified"] = (
                    verification_result["verification_score"] >= 7.0
                )

            # Generate recommendations
            if verification_result["verification_score"] < 6.0:
                verification_result["recommendations"].append(
                    "Consider finding additional sources to verify this information"
                )

            if verification_result["conflicts"]:
                verification_result["recommendations"].append(
                    "Review conflicting information from other sources"
                )

            if domain_score < 5.0:
                verification_result["recommendations"].append(
                    "Exercise caution due to low domain credibility"
                )

            return verification_result

        except Exception as e:
            logger.error(f"Error verifying citation credibility: {e}")
            return {"error": str(e), "verified": False}

    async def generate_bibliography(
        self, citation_ids: List[str], format_style: str = "apa"
    ) -> Dict:
        """
        Generate a formatted bibliography from a list of citation IDs.

        Args:
            citation_ids: List of citation IDs to include
            format_style: Citation format style (apa, mla, chicago, simple)

        Returns:
            Dictionary containing formatted bibliography
        """
        try:
            if format_style not in self.citation_formats:
                format_style = "apa"

            bibliography_entries = []
            missing_citations = []
            reliability_summary = {"high": 0, "medium": 0, "low": 0}

            for citation_id in citation_ids:
                if citation_id not in self.citations_cache:
                    missing_citations.append(citation_id)
                    continue

                citation = self.citations_cache[citation_id]

                # Get formatted citation
                formatted_citation = citation.get("citation_formats", {}).get(
                    format_style, citation["title"]
                )

                # Categorize reliability
                reliability_score = citation.get("reliability_score", 0)
                if reliability_score >= 8.0:
                    reliability_summary["high"] += 1
                    reliability_category = "high"
                elif reliability_score >= 5.0:
                    reliability_summary["medium"] += 1
                    reliability_category = "medium"
                else:
                    reliability_summary["low"] += 1
                    reliability_category = "low"

                bibliography_entries.append(
                    {
                        "id": citation_id,
                        "formatted_citation": formatted_citation,
                        "reliability_score": reliability_score,
                        "reliability_category": reliability_category,
                        "url": citation["url"],
                        "access_date": datetime.now().strftime("%Y-%m-%d"),
                    }
                )

            # Sort by title for consistency
            bibliography_entries.sort(key=lambda x: x["formatted_citation"].lower())

            return {
                "bibliography": bibliography_entries,
                "format_style": format_style,
                "total_sources": len(bibliography_entries),
                "reliability_summary": reliability_summary,
                "missing_citations": missing_citations,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error generating bibliography: {e}")
            return {"error": str(e), "bibliography": []}

    def _generate_citation_id(self, url: str, title: str) -> str:
        """Generate a unique citation ID."""
        content = f"{url}:{title}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def _detect_duplicates(
        self, citation_id: str, url: str, title: str, content: str
    ) -> Optional[Dict]:
        """Detect duplicate sources."""
        try:
            # Check URL duplicates
            for cached_id, cached_citation in self.citations_cache.items():
                if cached_citation["url"] == url:
                    return {
                        "type": "url_duplicate",
                        "original_id": cached_id,
                        "duplicate_url": url,
                    }

            # Check title similarity
            for cached_id, cached_citation in self.citations_cache.items():
                title_similarity = self._calculate_text_similarity(
                    title, cached_citation["title"]
                )
                if title_similarity > 0.9:
                    return {
                        "type": "title_duplicate",
                        "original_id": cached_id,
                        "similarity_score": title_similarity,
                    }

            return None

        except Exception as e:
            logger.error(f"Error detecting duplicates: {e}")
            return None

    async def _analyze_source_reliability(
        self, url: str, title: str, content: str
    ) -> Dict:
        """Analyze the reliability of a source."""
        try:
            domain = urlparse(url).netloc.lower()

            # Base domain scoring
            if any(
                edu_domain in domain for edu_domain in self.high_reliability_domains
            ):
                domain_score = 9.0
            elif any(
                med_domain in domain for med_domain in self.medium_reliability_domains
            ):
                domain_score = 7.0
            elif domain.endswith(".edu") or domain.endswith(".gov"):
                domain_score = 8.5
            elif domain.endswith(".org"):
                domain_score = 6.5
            else:
                domain_score = 5.0

            # Content quality factors
            content_factors = []

            # Check for academic indicators
            academic_indicators = [
                "doi:",
                "pmid:",
                "abstract",
                "methodology",
                "references",
                "peer review",
            ]
            academic_score = sum(
                1
                for indicator in academic_indicators
                if indicator.lower() in content.lower()
            )
            content_factors.append(
                ("academic_indicators", min(academic_score * 1.5, 10.0))
            )

            # Check for professional indicators
            professional_indicators = ["author", "publication date", "editor", "source"]
            professional_score = sum(
                1
                for indicator in professional_indicators
                if indicator.lower() in content.lower()
            )
            content_factors.append(
                ("professional_indicators", min(professional_score * 2.0, 10.0))
            )

            # Check content length and depth
            content_length = len(content)
            if content_length > 2000:
                length_score = 8.0
            elif content_length > 1000:
                length_score = 6.0
            elif content_length > 500:
                length_score = 4.0
            else:
                length_score = 2.0
            content_factors.append(("content_depth", length_score))

            # Calculate overall reliability score
            content_avg = (
                sum(score for _, score in content_factors) / len(content_factors)
                if content_factors
                else 5.0
            )
            overall_score = domain_score * 0.6 + content_avg * 0.4

            return {
                "score": round(overall_score, 1),
                "factors": {
                    "domain_score": domain_score,
                    "content_factors": content_factors,
                    "domain": domain,
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing source reliability: {e}")
            return {"score": 5.0, "factors": {"error": str(e)}}

    async def _extract_metadata(self, source: Dict) -> Dict:
        """Extract metadata from source."""
        try:
            url = source.get("url", "")
            domain = urlparse(url).netloc

            metadata = {
                "url": url,
                "domain": domain,
                "title": source.get("title", "Untitled"),
                "extraction_date": datetime.now().isoformat(),
                "content_length": len(source.get("content", "")),
                "language": "en",  # Default, could be enhanced with detection
            }

            # Try to extract publication date from content
            content = source.get("content", "")
            date_pattern = (
                r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{1,2}\s+\w+\s+\d{4})"
            )
            date_match = re.search(date_pattern, content)
            if date_match:
                metadata["publication_date"] = date_match.group(1)

            # Try to extract author information
            author_patterns = [
                r"by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
                r"author[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
                r"written\s+by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            ]

            for pattern in author_patterns:
                author_match = re.search(pattern, content, re.IGNORECASE)
                if author_match:
                    metadata["author"] = author_match.group(1)
                    break

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {
                "url": source.get("url", ""),
                "title": source.get("title", "Untitled"),
            }

    async def _generate_citation_formats(self, metadata: Dict) -> Dict:
        """Generate different citation formats."""
        try:
            formats = {}

            # Extract common fields
            title = metadata.get("title", "Untitled")
            url = metadata.get("url", "")
            domain = metadata.get("domain", "")
            author = metadata.get("author", "")
            pub_date = metadata.get(
                "publication_date", datetime.now().strftime("%Y-%m-%d")
            )
            access_date = datetime.now().strftime("%Y-%m-%d")

            # Simple format
            formats["simple"] = f"{title} - {domain} ({pub_date})"

            # APA format
            author_apa = f"{author}. " if author else ""
            year = pub_date.split("-")[0] if "-" in pub_date else pub_date[:4]
            formats["apa"] = f"{author_apa}({year}). {title}. Retrieved from {url}"

            # MLA format
            author_mla = f"{author}. " if author else ""
            formats["mla"] = f'{author_mla}"{title}." {domain}, {pub_date}, {url}.'

            # Chicago format
            author_chicago = f"{author}. " if author else ""
            formats["chicago"] = (
                f'{author_chicago}"{title}." {domain}. Accessed {access_date}. {url}.'
            )

            return formats

        except Exception as e:
            logger.error(f"Error generating citation formats: {e}")
            return {"simple": metadata.get("title", "Untitled")}

    async def _analyze_source_connections(
        self, citation_id: str, url: str, content: str, existing_citations: List[Dict]
    ) -> List[Dict]:
        """Analyze connections between sources."""
        try:
            connections = []

            # Check for URL references in content
            url_pattern = r'https?://[^\s<>"]+'
            found_urls = re.findall(url_pattern, content)

            for found_url in found_urls[:5]:  # Limit to first 5 found URLs
                for existing_citation in existing_citations:
                    if existing_citation["url"] == found_url:
                        connections.append(
                            {
                                "type": "url_reference",
                                "target_citation_id": existing_citation["id"],
                                "relationship": "references",
                            }
                        )

            # Check for topic similarity with existing citations
            for existing_citation in existing_citations:
                similarity = self._calculate_text_similarity(
                    content, existing_citation.get("title", "")
                )
                if similarity > 0.3:
                    connections.append(
                        {
                            "type": "topic_similarity",
                            "target_citation_id": existing_citation["id"],
                            "similarity_score": similarity,
                            "relationship": "similar_topic",
                        }
                    )

            return connections

        except Exception as e:
            logger.error(f"Error analyzing source connections: {e}")
            return []

    async def _analyze_citation_patterns(
        self, citations: List[Dict], context: str
    ) -> Dict:
        """Analyze patterns in the citation collection."""
        try:
            patterns = {
                "domain_distribution": {},
                "reliability_distribution": {"high": 0, "medium": 0, "low": 0},
                "temporal_pattern": {},
                "author_frequency": {},
                "common_topics": [],
            }

            for citation in citations:
                # Domain distribution
                domain = citation.get("metadata", {}).get("domain", "unknown")
                patterns["domain_distribution"][domain] = (
                    patterns["domain_distribution"].get(domain, 0) + 1
                )

                # Reliability distribution
                reliability = citation.get("reliability_score", 0)
                if reliability >= 8.0:
                    patterns["reliability_distribution"]["high"] += 1
                elif reliability >= 5.0:
                    patterns["reliability_distribution"]["medium"] += 1
                else:
                    patterns["reliability_distribution"]["low"] += 1

                # Author frequency
                author = citation.get("metadata", {}).get("author", "")
                if author:
                    patterns["author_frequency"][author] = (
                        patterns["author_frequency"].get(author, 0) + 1
                    )

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing citation patterns: {e}")
            return {}

    async def _generate_citation_summary(
        self, citations: List[Dict], reliability_analysis: Dict
    ) -> Dict:
        """Generate a summary of the citation collection."""
        try:
            if not citations:
                return {"message": "No citations to summarize"}

            total_sources = len(citations)
            avg_reliability = (
                sum(c.get("reliability_score", 0) for c in citations) / total_sources
            )

            # Top domains
            domain_counts = {}
            for citation in citations:
                domain = citation.get("metadata", {}).get("domain", "unknown")
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            top_domains = sorted(
                domain_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]

            summary = {
                "total_sources": total_sources,
                "average_reliability": round(avg_reliability, 1),
                "top_domains": top_domains,
                "reliability_breakdown": {
                    "high_quality": len(
                        [c for c in citations if c.get("reliability_score", 0) >= 8.0]
                    ),
                    "medium_quality": len(
                        [
                            c
                            for c in citations
                            if 5.0 <= c.get("reliability_score", 0) < 8.0
                        ]
                    ),
                    "low_quality": len(
                        [c for c in citations if c.get("reliability_score", 0) < 5.0]
                    ),
                },
                "recommendations": [],
            }

            # Generate recommendations
            if avg_reliability < 6.0:
                summary["recommendations"].append(
                    "Consider finding higher quality sources"
                )

            if summary["reliability_breakdown"]["low_quality"] > total_sources * 0.3:
                summary["recommendations"].append(
                    "Review and verify low-quality sources"
                )

            if len(set(domain_counts.keys())) < 3:
                summary["recommendations"].append(
                    "Diversify sources across more domains"
                )

            return summary

        except Exception as e:
            logger.error(f"Error generating citation summary: {e}")
            return {"error": str(e)}

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity score."""
        try:
            # Simple word-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            return len(intersection) / len(union) if union else 0.0

        except Exception:
            return 0.0

    async def _check_domain_credibility(self, url: str) -> float:
        """Check the credibility of a domain."""
        try:
            domain = urlparse(url).netloc.lower()

            if any(
                high_domain in domain for high_domain in self.high_reliability_domains
            ):
                return 9.0
            elif any(
                med_domain in domain for med_domain in self.medium_reliability_domains
            ):
                return 7.0
            elif domain.endswith(".edu") or domain.endswith(".gov"):
                return 8.5
            elif domain.endswith(".org"):
                return 6.5
            else:
                return 5.0

        except Exception:
            return 3.0

    async def _check_content_authenticity(self, citation: Dict) -> Dict:
        """Check content authenticity indicators."""
        try:
            url = citation.get("url", "")
            title = citation.get("title", "")

            authenticity_score = 5.0
            details = []

            # Check for HTTPS
            if url.startswith("https://"):
                authenticity_score += 1.0
                details.append("Secure HTTPS connection")

            # Check for proper title
            if len(title) > 10 and not title.lower().startswith("untitled"):
                authenticity_score += 1.0
                details.append("Descriptive title present")

            # Check URL structure
            if not any(
                spam_indicator in url.lower()
                for spam_indicator in ["bit.ly", "tinyurl", "spam"]
            ):
                authenticity_score += 1.0
                details.append("Clean URL structure")

            return {"score": min(authenticity_score, 10.0), "details": details}

        except Exception as e:
            return {"score": 3.0, "details": [f"Error checking authenticity: {e}"]}

    async def _cross_reference_sources(
        self, citation: Dict, cross_check_sources: List[Dict]
    ) -> Dict:
        """Cross-reference a citation with other sources."""
        try:
            matches = []
            conflicts = []
            consistency_score = 5.0

            citation_title = citation.get("title", "").lower()

            for source in cross_check_sources:
                source_title = source.get("title", "").lower()
                similarity = self._calculate_text_similarity(
                    citation_title, source_title
                )

                if similarity > 0.5:
                    matches.append(
                        {
                            "source_url": source.get("url", ""),
                            "similarity": similarity,
                            "type": "title_match",
                        }
                    )
                    consistency_score += 1.0

                # Check for conflicting information (simplified)
                citation_content = citation.get("content", "").lower()
                source_content = source.get("content", "").lower()

                # Look for contradictory keywords
                contradiction_pairs = [
                    ["true", "false"],
                    ["yes", "no"],
                    ["increase", "decrease"],
                    ["positive", "negative"],
                    ["effective", "ineffective"],
                ]

                for pos_word, neg_word in contradiction_pairs:
                    if pos_word in citation_content and neg_word in source_content:
                        conflicts.append(
                            {
                                "source_url": source.get("url", ""),
                                "conflict_type": "contradictory_keywords",
                                "keywords": [pos_word, neg_word],
                            }
                        )
                        consistency_score -= 0.5

            return {
                "matches": matches,
                "conflicts": conflicts,
                "consistency_score": max(min(consistency_score, 10.0), 0.0),
            }

        except Exception as e:
            logger.error(f"Error cross-referencing sources: {e}")
            return {"matches": [], "conflicts": [], "consistency_score": 5.0}
