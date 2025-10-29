"""
Multi-strategy selector generator for robust element finding.

This module generates multiple fallback strategies to find elements on a page,
reducing dependence on AI and making workflows more deterministic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SelectorStrategy:
	"""A single selector strategy with priority and metadata."""

	type: str  # Strategy type: 'id', 'css_attr', 'text_exact', 'aria', etc.
	value: str  # The selector value or matching text
	priority: int  # Lower = try first (1 is highest priority)
	metadata: Dict[str, Any] = field(default_factory=dict)  # Extra info for matching

	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization."""
		return {
			'type': self.type,
			'value': self.value,
			'priority': self.priority,
			'metadata': self.metadata,
		}

	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'SelectorStrategy':
		"""Create from dictionary."""
		return cls(
			type=data['type'],
			value=data['value'],
			priority=data['priority'],
			metadata=data.get('metadata', {}),
		)


class SelectorGenerator:
	"""
	Generate multiple robust selector strategies from element data.

	This class takes element data captured during workflow recording and generates
	a prioritized list of strategies to find that element during execution.

	The strategies are ordered by reliability:
	1. ID selectors (most stable)
	2. Data attributes (very stable)
	3. Name attributes (stable for forms)
	4. Exact text match (good for links/buttons)
	5. ARIA labels (accessibility-based)
	6. Role + text (semantic HTML)
	7. Placeholder (for inputs)
	8. Class + text combination
	9. Fuzzy text match (resilient to small changes)
	10. Direct CSS/xpath (fallback)
	"""

	def generate_strategies(self, element_data: Dict[str, Any]) -> List[SelectorStrategy]:
		"""
		Generate SEMANTIC-ONLY selector strategies from captured element data.

		No CSS selectors, no xpaths - only human-readable semantic strategies.

		Args:
		    element_data: Dictionary containing:
		        - tag_name: str (e.g., 'a', 'button', 'input')
		        - text: str (visible text content)
		        - attributes: Dict[str, str] (element attributes)

		Returns:
		    List of SelectorStrategy objects, ordered by priority

		Example:
		    >>> generator = SelectorGenerator()
		    >>> strategies = generator.generate_strategies(
		    ...     {
		    ...         'tag_name': 'button',
		    ...         'text': 'Submit',
		    ...         'attributes': {'aria-label': 'Submit form'},
		    ...     }
		    ... )
		    >>> # Returns: text_exact, role_text, aria_label, text_fuzzy
		"""
		strategies = []
		tag = element_data.get('tag_name', '').lower()
		text = element_data.get('text', '').strip()
		attrs = element_data.get('attributes', {})

		# Strategy 1: Exact text match (highest priority - most reliable)
		if text:
			strategies.append(
				SelectorStrategy(
					type='text_exact',
					value=text,
					priority=1,
					metadata={'tag': tag},
				)
			)

		# Strategy 2: Role + text (semantic HTML)
		role = self._infer_role(tag, attrs)
		if role and text:
			strategies.append(
				SelectorStrategy(
					type='role_text',
					value=text,
					priority=2,
					metadata={'role': role, 'tag': tag},
				)
			)

		# Strategy 3: ARIA label (accessibility-based)
		if 'aria-label' in attrs and attrs['aria-label']:
			strategies.append(
				SelectorStrategy(
					type='aria_label',
					value=attrs['aria-label'],
					priority=3,
					metadata={'tag': tag},
				)
			)

		# Strategy 4: Placeholder (for input fields)
		if 'placeholder' in attrs and attrs['placeholder']:
			strategies.append(
				SelectorStrategy(
					type='placeholder',
					value=attrs['placeholder'],
					priority=4,
					metadata={'tag': tag},
				)
			)

		# Strategy 5: Title attribute (tooltip text)
		if 'title' in attrs and attrs['title']:
			strategies.append(
				SelectorStrategy(
					type='title',
					value=attrs['title'],
					priority=5,
					metadata={'tag': tag},
				)
			)

		# Strategy 6: Alt text (for images)
		if 'alt' in attrs and attrs['alt']:
			strategies.append(
				SelectorStrategy(
					type='alt_text',
					value=attrs['alt'],
					priority=6,
					metadata={'tag': tag},
				)
			)

		# Strategy 7: Fuzzy text match (fallback - handles typos/variations)
		if text and len(text) > 3:  # Only for meaningful text
			strategies.append(
				SelectorStrategy(
					type='text_fuzzy',
					value=text,
					priority=7,
					metadata={'threshold': 0.8, 'tag': tag},
				)
			)

		# Sort by priority (lower number = higher priority)
		strategies.sort(key=lambda s: s.priority)

		return strategies

	def _infer_role(self, tag: str, attrs: Dict[str, Any]) -> Optional[str]:
		"""
		Infer semantic role from HTML tag and attributes.

		Args:
		    tag: HTML tag name (e.g., 'button', 'a', 'input')
		    attrs: Element attributes

		Returns:
		    Semantic role string or None
		"""
		# Explicit role attribute takes precedence
		if 'role' in attrs:
			return attrs['role']

		# Infer from HTML tag
		role_map = {
			'button': 'button',
			'a': 'link',
			'input': 'textbox',
			'textarea': 'textbox',
			'select': 'combobox',
			'h1': 'heading',
			'h2': 'heading',
			'h3': 'heading',
			'h4': 'heading',
			'h5': 'heading',
			'h6': 'heading',
			'img': 'img',
			'table': 'table',
			'ul': 'list',
			'ol': 'list',
			'nav': 'navigation',
		}

		# Special case for input types
		if tag == 'input' and 'type' in attrs:
			input_type = attrs['type'].lower()
			if input_type == 'checkbox':
				return 'checkbox'
			elif input_type == 'radio':
				return 'radio'
			elif input_type == 'submit':
				return 'button'

		return role_map.get(tag)

	def _escape_quotes(self, value: str) -> str:
		"""Escape quotes in CSS selector values."""
		return value.replace("'", "\\'").replace('"', '\\"')

	def generate_strategies_dict(self, element_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""
		Generate strategies and return as list of dictionaries for JSON serialization.

		Args:
		    element_data: Element data dictionary

		Returns:
		    List of strategy dictionaries
		"""
		strategies = self.generate_strategies(element_data)
		return [s.to_dict() for s in strategies]

	def get_summary(self, strategies: List[SelectorStrategy]) -> str:
		"""
		Get a human-readable summary of strategies.

		Args:
		    strategies: List of selector strategies

		Returns:
		    Summary string

		Example:
		    >>> generator = SelectorGenerator()
		    >>> strategies = generator.generate_strategies({...})
		    >>> print(generator.get_summary(strategies))
		    Generated 5 selector strategies:
		      1. [priority 1] id: #submit-btn
		      2. [priority 4] text_exact: "Submit"
		      ...
		"""
		lines = [f'Generated {len(strategies)} selector strategies:']
		for i, s in enumerate(strategies[:5], 1):  # Show first 5
			value_preview = s.value[:50] + '...' if len(s.value) > 50 else s.value
			lines.append(f'  {i}. [priority {s.priority}] {s.type}: {value_preview}')
		if len(strategies) > 5:
			lines.append(f'  ... and {len(strategies) - 5} more')
		return '\n'.join(lines)
