"""
Example usage of the AI Meeting Assistant
Demonstrates how to use the application programmatically
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import MeetingAssistant


def example_1_process_existing_file():
    """Example 1: Process an existing audio file"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Process Existing Audio File")
    print("="*80 + "\n")

    assistant = MeetingAssistant()

    # Replace with your actual audio file path
    audio_file = "data/sample_meeting.mp3"

    if Path(audio_file).exists():
        report = assistant.process_meeting(
            audio_file_path=audio_file,
            meeting_title="Q1 Planning Meeting",
            language="en"
        )

        print("\n" + "-"*80)
        print("Processing complete! Report saved.")
        print("-"*80)
    else:
        print(f"Audio file not found: {audio_file}")
        print("Please provide a valid audio file path.")


def example_2_analyze_transcription():
    """Example 2: Analyze a transcription directly"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Analyze Transcription Directly")
    print("="*80 + "\n")

    assistant = MeetingAssistant()

    sample_transcription = """
    John: Good morning everyone. Let's start our quarterly review meeting.
    We need to discuss our Q1 performance, upcoming projects, and budget allocation.

    Sarah: Thanks John. Q1 was very productive. We completed three major projects on time.
    The team worked really well together. However, we did face some resource constraints.

    John: That's good to hear. What about the Q2 roadmap?

    Mike: We have three major initiatives planned for Q2. First is the dashboard redesign,
    second is API performance optimization, and third is implementing the new analytics system.
    We need to finalize the budget by end of this week.

    Sarah: I agree. For the dashboard project, we should allocate additional resources.
    I recommend hiring two contractors for three months.

    John: Good idea. Let's make that decision final. Sarah, can you prepare a budget proposal?
    Mike, please outline the timeline for all three projects by Friday.

    Mike: Will do. Should I include risk assessments as well?

    John: Yes, definitely. That would be very helpful.
    Sarah: I'll also need the detailed resource breakdown for budgeting purposes.

    John: Perfect. Let's reconvene next week to review the proposals.
    Meeting adjourned.
    """

    # Analyze the transcription
    summary = assistant.summarizer.summarize(sample_transcription)
    key_points = assistant.summarizer.extract_key_points(sample_transcription, num_points=5)
    action_items = assistant.summarizer.extract_action_items(sample_transcription)
    participants = assistant.summarizer.extract_participants(sample_transcription)

    # Display results
    print("SUMMARY:")
    print("-" * 40)
    print(summary)

    print("\n\nKEY POINTS:")
    print("-" * 40)
    for i, point in enumerate(key_points, 1):
        print(f"{i}. {point}")

    print("\n\nACTION ITEMS:")
    print("-" * 40)
    print(action_items)

    print("\n\nPARTICIPANTS:")
    print("-" * 40)
    for participant in participants:
        print(f"• {participant}")


def example_3_save_custom_report():
    """Example 3: Save a custom report"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Save Custom Report")
    print("="*80 + "\n")

    assistant = MeetingAssistant()

    sample_report = {
        'summary': 'This was a productive quarterly review meeting focusing on Q1 achievements and Q2 planning.',
        'key_points': [
            'Q1 was productive with three major projects completed on time',
            'Q2 has three major initiatives: dashboard redesign, API optimization, and analytics',
            'Need to hire two contractors for dashboard project',
            'Budget proposal and timeline required by Friday',
            'Follow-up meeting scheduled for next week'
        ],
        'action_items': '1. Sarah to prepare budget proposal for Q2 initiatives\n2. Mike to outline timeline for three Q2 projects and include risk assessments\n3. Sarah to provide detailed resource breakdown for budgeting',
        'participants': ['John', 'Sarah', 'Mike'],
        'transcription': 'Full transcription text would go here...'
    }

    # Save reports
    text_path = assistant.report_generator.save_report_to_file(
        "Quarterly_Review_Meeting",
        sample_report
    )

    json_path = assistant.report_generator.save_report_as_json(
        "Quarterly_Review_Meeting",
        sample_report
    )

    print(f"✓ Text report saved to: {text_path}")
    print(f"✓ JSON report saved to: {json_path}")


def example_4_view_meeting_history():
    """Example 4: View meeting history"""
    print("\n" + "="*80)
    print("EXAMPLE 4: View Meeting History")
    print("="*80 + "\n")

    assistant = MeetingAssistant()

    history = assistant.file_manager.get_meeting_history()

    if history:
        print(f"Found {len(history)} meetings in history:\n")
        for i, meeting in enumerate(history, 1):
            print(f"{i}. {meeting['title']}")
            print(f"   File: {meeting['filename']}")
            print(f"   Date: {meeting['timestamp']}")
            if meeting.get('participants'):
                print(f"   Participants: {', '.join(meeting['participants'])}")
            print()
    else:
        print("No meetings in history yet.")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AI MEETING ASSISTANT - Usage Examples")
    print("="*80)

    print("\nSelect an example to run:")
    print("1. Process existing audio file")
    print("2. Analyze transcription directly (sample)")
    print("3. Save custom report")
    print("4. View meeting history")
    print("0. Run all examples")

    choice = input("\nEnter choice (0-4): ").strip()

    try:
        if choice == '1':
            example_1_process_existing_file()
        elif choice == '2':
            example_2_analyze_transcription()
        elif choice == '3':
            example_3_save_custom_report()
        elif choice == '4':
            example_4_view_meeting_history()
        elif choice == '0':
            example_2_analyze_transcription()
            example_3_save_custom_report()
            example_4_view_meeting_history()
        else:
            print("Invalid choice")
    except KeyError as e:
        print(f"\nError: {e}")
        print("Please set OPENAI_API_KEY in your .env file")
    except Exception as e:
        print(f"\nError: {e}")
