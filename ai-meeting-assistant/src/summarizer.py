"""
Meeting Summarization Module
Extracts key points, decisions, and generates summaries from meeting transcriptions

Uses base-provider for flexible LLM configuration
"""

from base_provider import Message


class MeetingSummarizer:
    """Summarizes meeting transcriptions and extracts key information"""

    def __init__(self, provider=None):
        """
        Initialize the summarizer with an LLM provider

        Args:
            provider (LLMProvider, optional): LLM provider instance.
                If not provided, uses get_llm_provider() from config module
        """
        self.provider = provider
        if self.provider is None:
            from .config import get_llm_provider
            self.provider = get_llm_provider()

        self.model = self.provider.model

    def summarize(self, transcription_text, max_tokens=500, style='concise'):
        """
        Generate a concise summary of the meeting

        Args:
            transcription_text (str): Full transcription text
            max_tokens (int): Maximum tokens for the summary
            style (str): Summary style - 'concise', 'detailed', or 'bullet'

        Returns:
            str: Meeting summary
        """
        prompt = self._build_summary_prompt(transcription_text, style)

        response = self.provider.complete(
            messages=[
                Message("system", "You are an expert business meeting analyst. Provide clear, concise summaries of meetings."),
                Message("user", prompt)
            ],
            max_tokens=max_tokens,
            temperature=0.5
        )

        return response.content

    def extract_key_points(self, transcription_text, num_points=5):
        """
        Extract the most important points from the meeting

        Args:
            transcription_text (str): Full transcription text
            num_points (int): Number of key points to extract

        Returns:
            list: List of key points
        """
        prompt = f"""
Extract exactly {num_points} key points from this meeting transcription.
Format each point as a single clear sentence.
Focus on decisions, action items, and important information.

Transcription:
{transcription_text}

Key Points:
"""

        response = self.provider.complete(
            messages=[
                Message("system", "You are an expert at identifying key points in business meetings."),
                Message("user", prompt)
            ],
            max_tokens=300,
            temperature=0.5
        )

        # Parse the response into a list
        content = response.content
        points = [p.strip() for p in content.split('\n') if p.strip() and p[0].isdigit()]

        return points

    def extract_action_items(self, transcription_text):
        """
        Extract action items and assignments from the meeting

        Args:
            transcription_text (str): Full transcription text

        Returns:
            list: List of action items with owner information
        """
        prompt = f"""
Extract all action items from this meeting transcription.
For each action item, identify:
1. The action to be taken
2. Who is responsible (if mentioned)
3. Any deadline mentioned

Format as bullet points with clear ownership.

Transcription:
{transcription_text}

Action Items:
"""

        response = self.provider.complete(
            messages=[
                Message("system", "You are skilled at extracting action items and assignments from business meetings."),
                Message("user", prompt)
            ],
            max_tokens=400,
            temperature=0.5
        )

        return response.content

    def extract_participants(self, transcription_text):
        """
        Extract and identify meeting participants

        Args:
            transcription_text (str): Full transcription text

        Returns:
            list: List of participant names
        """
        prompt = f"""
Identify all people mentioned as participants in this meeting.
Extract their names only, without titles.
Return one name per line.

Transcription:
{transcription_text}

Participants:
"""

        response = self.provider.complete(
            messages=[
                Message("system", "You are skilled at identifying meeting participants from transcriptions."),
                Message("user", prompt)
            ],
            max_tokens=150,
            temperature=0.3
        )

        # Parse the response into a list
        content = response.content
        participants = [p.strip() for p in content.split('\n') if p.strip()]

        return participants

    def generate_full_report(self, transcription_text):
        """
        Generate a comprehensive meeting report

        Args:
            transcription_text (str): Full transcription text

        Returns:
            dict: Comprehensive meeting report
        """
        print("Generating comprehensive meeting report...")

        report = {
            'summary': self.summarize(transcription_text, max_tokens=500, style='concise'),
            'key_points': self.extract_key_points(transcription_text, num_points=5),
            'action_items': self.extract_action_items(transcription_text),
            'participants': self.extract_participants(transcription_text)
        }

        return report

    def _build_summary_prompt(self, transcription_text, style):
        """Build the prompt for summarization based on style"""
        styles = {
            'concise': "Provide a brief 2-3 sentence summary of the meeting.",
            'detailed': "Provide a detailed summary of the meeting, covering all major topics discussed.",
            'bullet': "Provide a bullet-point summary of the meeting, with 3-5 main topics."
        }

        instruction = styles.get(style, styles['concise'])

        return f"""{instruction}

Transcription:
{transcription_text}

Summary:
"""
